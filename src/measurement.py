from __future__ import annotations
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Union

@dataclass
class ScaleModel:
    px_per_cm: float

PointLike = Union[Tuple[float, float], np.ndarray]

def set_scale_by_two_points(p1: PointLike, p2: PointLike, real_distance_cm: float) -> ScaleModel:
    """Fija la escala (px/cm) usando dos puntos y una distancia real conocida."""
    p1a = np.asarray(p1, dtype=np.float32)
    p2a = np.asarray(p2, dtype=np.float32)
    dpx = float(np.linalg.norm(p1a - p2a))
    px_per_cm = float(dpx / float(real_distance_cm))
    return ScaleModel(px_per_cm=px_per_cm)

def measure_distance(p1: PointLike, p2: PointLike, scale: ScaleModel) -> float:
    """Devuelve la distancia estimada en centímetros entre dos puntos (píxeles) dado el modelo de escala."""
    p1a = np.asarray(p1, dtype=np.float32)
    p2a = np.asarray(p2, dtype=np.float32)
    dpx = float(np.linalg.norm(p1a - p2a))
    return float(dpx / float(scale.px_per_cm))

def interactive_pick_points(image: np.ndarray, npoints: int = 2, window_name: str = "pick", auto_close: bool = True) -> List[Tuple[float,float]]:
    """Selector simple de puntos con OpenCV (compatible con cuadernos).
    - Click izquierdo: agrega punto
    - Tecla 'r': reinicia
    - ENTER/'q'/ESC: finaliza si ya hay npoints
    - Si ``auto_close`` es True, finaliza automáticamente al alcanzar ``npoints``
    """
    clone = image.copy()
    pts: List[Tuple[float,float]] = []
    done = False

    def on_mouse(event, x, y, flags, param):
        nonlocal clone, pts, done
        if event == cv.EVENT_LBUTTONDOWN and len(pts) < npoints:
            pts.append((float(x), float(y)))
            cv.circle(clone, (x, y), 4, (0,255,0), -1)
            cv.putText(clone, str(len(pts)), (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)
            cv.imshow(window_name, clone)
            if auto_close and len(pts) >= npoints:
                done = True

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    try:
        # Trae la ventana al frente en Windows si está soportado
        cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv.imshow(window_name, clone)
    cv.setMouseCallback(window_name, on_mouse)  # type: ignore[attr-defined]

    while True:
        if done:
            break
        k = cv.waitKey(20) & 0xFF
        # ENTER (13), 'q' o ESC (27)
        if (k in (13, ord('q'), 27)) and len(pts) >= npoints:
            break
        if k == ord('r'):
            pts.clear()
            clone = image.copy()
            cv.imshow(window_name, clone)
        # Si el usuario cierra la ventana manualmente
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    cv.destroyWindow(window_name)
    return pts
