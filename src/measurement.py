from __future__ import annotations
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class ScaleModel:
    px_per_cm: float

def set_scale_by_two_points(p1: Tuple[float,float], p2: Tuple[float,float], real_distance_cm: float) -> ScaleModel:
    """Fija la escala (px/cm) usando dos puntos y una distancia real conocida."""
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    dpx = np.linalg.norm(p1 - p2)
    px_per_cm = dpx / real_distance_cm
    return ScaleModel(px_per_cm=px_per_cm)

def measure_distance(p1: Tuple[float,float], p2: Tuple[float,float], scale: ScaleModel) -> float:
    """Devuelve la distancia estimada en centímetros entre dos puntos (píxeles) dado el modelo de escala."""
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    dpx = np.linalg.norm(p1 - p2)
    return float(dpx / scale.px_per_cm)

def interactive_pick_points(image: np.ndarray, npoints: int = 2) -> List[Tuple[float,float]]:
    """Herramienta simple para elegir puntos con mouse en una ventana de OpenCV.
    - Click izquierdo: agrega punto
    - Tecla 'r': reinicia
    - Tecla 'q' o ESC: sale si ya hay npoints
    """
    clone = image.copy()
    pts: List[Tuple[float,float]] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal clone, pts
        if event == cv.EVENT_LBUTTONDOWN and len(pts) < npoints:
            pts.append((float(x), float(y)))
            cv.circle(clone, (x,y), 4, (0,255,0), -1)
            cv.imshow("pick", clone)

    cv.namedWindow("pick", cv.WINDOW_NORMAL)
    cv.setMouseCallback("pick", on_mouse)
    cv.imshow("pick", clone)
    while True:
        k = cv.waitKey(10) & 0xFF
        if (k == ord('q') or k == 27) and len(pts) >= npoints:
            break
        if k == ord('r'):
            pts.clear()
            clone = image.copy()
            cv.imshow("pick", clone)
    cv.destroyWindow("pick")
    return pts
