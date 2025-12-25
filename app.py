"""
Aplicación de escritorio para detección de objetos en tiempo real usando la cámara integrada y un
modelo ligero basado en MobileNet SSD.

Instrucciones de uso:
1) Instalar dependencias recomendadas:
   python -m pip install opencv-python numpy pillow requests
2) Ejecutar la aplicación:
   python app.py

Requisitos:
- Python 3.8 o superior.
- Cámara integrada funcional.
- Conexión a internet para la primera ejecución (descarga automática del modelo).
"""

from __future__ import annotations

import importlib.util
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# ---------------------------
# Utilidades de dependencias
# ---------------------------

def ensure_dependencies() -> None:
    """Verifica dependencias básicas y ofrece instalación automática si faltan."""
    required = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("requests", "requests"),
    ]
    missing = []
    for package, import_name in required:
        if importlib.util.find_spec(import_name) is None:
            missing.append(package)
    if not missing:
        return

    print("Dependencias faltantes detectadas:", ", ".join(missing))
    response = input("¿Desea instalarlas automáticamente ahora? [s/N]: ").strip().lower()
    if response != "s":
        print("Instale manualmente con: python -m pip install " + " ".join(missing))
        sys.exit(1)

    for package in missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"No se pudo instalar {package}. Intente manualmente.")
            sys.exit(1)


ensure_dependencies()

import cv2
import numpy as np
import requests
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox


# ---------------------------
# Datos y configuración
# ---------------------------

@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


MODEL_LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


MODEL_SPECS = {
    "mobilenet_ssd": {
        "name": "MobileNet SSD (VOC)",
        "description": "Modelo ligero en Caffe entrenado sobre VOC 20 clases.",
        "prototxt_urls": [
            # Copia mantenida por OpenCV; URL estable.
            "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/MobileNetSSD_deploy.prototxt",
            # Copia de respaldo del repositorio original.
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
        ],
        "weights_urls": [
            # Pesos originales liberados por los autores.
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
            # Espejo alternativo en el dataset público de OpenCV.
            "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/MobileNetSSD_deploy.caffemodel",
        ],
        "labels": MODEL_LABELS,
    },
}


# ---------------------------
# Clase de cámara
# ---------------------------

class CameraHandler:
    """Gestiona el acceso a la cámara y la captura de frames."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        """Intenta abrir la cámara seleccionada."""
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True

    def read(self) -> Tuple[bool, np.ndarray | None]:
        """Lee un frame de la cámara abierta."""
        if self.capture is None:
            return False, None
        return self.capture.read()

    def release(self) -> None:
        """Libera la cámara si está en uso."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None


# ---------------------------
# Clase de detección
# ---------------------------

class ObjectDetector:
    """Carga el modelo y realiza detección de objetos."""

    def __init__(self, model_name: str = "mobilenet_ssd", model_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.net: cv2.dnn_Net | None = None
        self.labels: List[str] = MODEL_LABELS

    def _download_file(self, urls: List[str], destination: Path) -> None:
        """Descarga un archivo intentando múltiples URLs como respaldo."""
        errors: list[str] = []
        for url in urls:
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                with open(destination, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                progress = downloaded * 100 / total
                                print(f"Descargando {destination.name}: {progress:.2f}% desde {url}", end="\r")
                print()
                return
            except requests.RequestException as error:
                errors.append(f"{url} -> {error}")
                continue

        raise ConnectionError("; ".join(errors))

    def ensure_model(self) -> None:
        """Verifica y descarga el modelo seleccionado si es necesario."""
        config = MODEL_SPECS.get(self.model_name)
        if not config:
            raise ValueError(f"Modelo {self.model_name} no soportado.")

        prototxt_path = self.model_dir / f"{self.model_name}.prototxt"
        weights_path = self.model_dir / f"{self.model_name}.caffemodel"

        if not prototxt_path.exists():
            try:
                self._download_file(config["prototxt_urls"], prototxt_path)
            except ConnectionError as error:
                raise ConnectionError(f"Error descargando prototxt: {error}") from error

        if not weights_path.exists():
            try:
                self._download_file(config["weights_urls"], weights_path)
            except ConnectionError as error:
                raise ConnectionError(f"Error descargando pesos: {error}") from error

        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.labels = config.get("labels", MODEL_LABELS)

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.4) -> List[Detection]:
        """Ejecuta detección sobre un frame y devuelve resultados."""
        if self.net is None:
            raise RuntimeError("El modelo no está cargado. Llame a ensure_model() primero.")

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results: List[Detection] = []
        h, w = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < confidence_threshold:
                continue
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")
            label_list = self.labels
            label_value = label_list[class_id] if class_id < len(label_list) else str(class_id)
            results.append(
                Detection(
                    label=label_value,
                    confidence=confidence,
                    box=(start_x, start_y, end_x, end_y),
                )
            )
        return results


# ---------------------------
# Clase de interfaz gráfica
# ---------------------------

class ApplicationUI:
    """Interfaz gráfica que coordina cámara, detección y visualización."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Detección de objetos en vivo")
        self.root.geometry("900x650")
        self.root.configure(bg="#1e1e1e")

        self.camera = CameraHandler()
        self.detector = ObjectDetector()
        self.video_running = False
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.available_models = list(MODEL_SPECS.keys())

        self._build_layout()

    def _build_layout(self) -> None:
        """Crea los elementos de la interfaz gráfica."""
        header = tk.Label(
            self.root,
            text="Aplicación de Detección de Objetos",
            font=("Segoe UI", 18, "bold"),
            bg="#1e1e1e",
            fg="#f0f0f0",
        )
        header.pack(pady=10)

        control_frame = tk.Frame(self.root, bg="#1e1e1e")
        control_frame.pack(pady=5)

        self.start_button = ttk.Button(control_frame, text="Iniciar", command=self.start_video)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Detener", command=self.stop_video, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        model_label = tk.Label(control_frame, text="Modelo:", bg="#1e1e1e", fg="#f0f0f0")
        model_label.grid(row=0, column=2, padx=5)

        self.model_var = tk.StringVar(value=self.available_models[0])
        self.model_selector = ttk.OptionMenu(control_frame, self.model_var, self.available_models[0], *self.available_models)
        self.model_selector.grid(row=0, column=3, padx=5)
        self.model_var.trace_add("write", lambda *_: self._on_model_change())

        model_info = self._describe_model(self.model_var.get())
        self.model_info_label = tk.Label(
            self.root,
            text=model_info,
            bg="#1e1e1e",
            fg="#cccccc",
            font=("Segoe UI", 9),
            wraplength=820,
            justify=tk.LEFT,
        )
        self.model_info_label.pack(pady=(0, 8))

        self.status_var = tk.StringVar(value="Modelo listo")
        status_label = tk.Label(self.root, textvariable=self.status_var, bg="#1e1e1e", fg="#a0e7a0")
        status_label.pack(pady=5)

        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.pack(pady=10)

        log_frame = tk.Frame(self.root, bg="#1e1e1e")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        log_label = tk.Label(log_frame, text="Logs", bg="#1e1e1e", fg="#f0f0f0")
        log_label.pack(anchor="w")

        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, bg="#121212", fg="#d0d0d0")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _describe_model(self, model_key: str) -> str:
        """Devuelve un texto descriptivo del modelo seleccionado."""
        spec = MODEL_SPECS.get(model_key)
        if not spec:
            return "Modelo no reconocido."
        primary_source = spec["weights_urls"][0]
        return f"Modelo activo: {spec['name']} · {spec['description']} · Fuente principal: {primary_source}"

    def _on_model_change(self) -> None:
        """Actualiza la descripción visible cuando el usuario elige otro modelo."""
        selected = self.model_var.get()
        self.model_info_label.configure(text=self._describe_model(selected))
        self.log(f"Modelo seleccionado: {selected}")

    def log(self, message: str) -> None:
        """Escribe un mensaje en el panel de logs."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def start_video(self) -> None:
        """Inicia la captura y detección de video."""
        if self.video_running:
            return

        self.status_var.set("Conectando cámara...")
        if not self.camera.open():
            self.status_var.set("No se encontró la cámara")
            messagebox.showerror("Cámara", "No se pudo acceder a la cámara. Verifique la conexión.")
            self.log("Error: cámara no disponible.")
            return

        self.detector.model_name = self.model_var.get()
        self.log(self._describe_model(self.detector.model_name))
        try:
            self.detector.ensure_model()
        except ConnectionError as error:
            self.status_var.set("Fallo al descargar modelo")
            messagebox.showerror("Modelo", f"No se pudo descargar el modelo: {error}")
            self.log(f"Error descargando modelo: {error}")
            self.camera.release()
            return
        except Exception as error:  # noqa: BLE001
            self.status_var.set("Fallo al cargar modelo")
            messagebox.showerror("Modelo", f"No se pudo cargar el modelo: {error}")
            self.log(f"Error cargando modelo: {error}")
            self.camera.release()
            return

        self.log("Cámara conectada y modelo cargado.")
        self.status_var.set("Detectando objetos")
        self.video_running = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

        threading.Thread(target=self._capture_loop, daemon=True).start()
        self._update_frame()

    def stop_video(self) -> None:
        """Detiene la captura de video."""
        if self.video_running:
            self.video_running = False
            self.camera.release()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Video detenido")
        self.log("Video detenido por el usuario.")

    def on_close(self) -> None:
        """Finaliza el video y cierra la aplicación."""
        self.stop_video()
        self.root.destroy()

    def _capture_loop(self) -> None:
        """Captura frames en un hilo separado para mantener la UI fluida."""
        while self.video_running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.status_var.set("Sin señal de cámara")
                self.log("Advertencia: no se pudo leer frame.")
                time.sleep(0.1)
                continue

            detections = self.detector.detect(frame)
            for detection in detections:
                start_x, start_y, end_x, end_y = detection.box
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                label = f"{detection.label}: {detection.confidence * 100:.1f}%"
                y_text = start_y - 10 if start_y - 10 > 10 else start_y + 10
                cv2.putText(frame, label, (start_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass

    def _update_frame(self) -> None:
        """Actualiza la imagen mostrada en la interfaz a intervalos regulares."""
        if not self.video_running:
            return

        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            self.root.after(30, self._update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((800, 450))
        imgtk = ImageTk.PhotoImage(image=image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(30, self._update_frame)


# ---------------------------
# Función principal
# ---------------------------

def main() -> None:
    """Punto de entrada de la aplicación."""
    root = tk.Tk()
    app = ApplicationUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
