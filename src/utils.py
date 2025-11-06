from __future__ import annotations
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def imread_color(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def draw_matches(img1, kps1, img2, kps2, matches, max_draw: int = 50):
    drawn = cv.drawMatches(img1, kps1, img2, kps2, matches[:max_draw], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return drawn

def synthetic_affine_pair(img: np.ndarray, angle_deg: float = 10.0, tx: float = 30.0, ty: float = 15.0, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera un par (img, img_transformada) y devuelve la matriz de transformación 3x3 (afin/proyectiva)."""
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w/2, h/2), angle_deg, scale)
    M[:,2] += (tx, ty)
    H = np.vstack([M, [0,0,1]])
    warped = cv.warpAffine(img, M, (w, h))
    return img, warped, H

def rmse(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean((A - B)**2)))

def show_image(img, title="", scale=1):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(5 * scale, 5 * scale))
    plt.title(title)
    plt.imshow(img)
    plt.axis("off")
    plt.show()