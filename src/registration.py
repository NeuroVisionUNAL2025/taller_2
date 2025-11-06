from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import Tuple, List

def estimate_homography(kp1, kp2, matches, ransac_thresh: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Estima H con RANSAC a partir de keypoints y matches.

    Returns
    -------
    H : (3,3) homografía
    mask : inliers (Nx1)
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H is None:
        raise RuntimeError("No se pudo estimar homografía; revisa correspondencias o parámetros.")
    return H, mask


def stitch_images_blend(images: List[np.ndarray], homographies: List[np.ndarray], blend: str = "feather") -> np.ndarray:
    """Une una lista de n imágenes usando n-1 homografías con transiciones suaves.

    Parámetros
    ----------
    images : lista de imágenes (BGR) [img0, img1, ..., img{n-1}]
    homographies : lista de homografías [H1, H2, ..., H{n-1}]
        Cada H_i mapea img_i -> img_0 (mismo sistema de referencia que img0).
    blend : tipo de blending. Solo 'feather' (feathering por distancia a borde).

    Returns
    -------
    panorama : imagen cosida con transiciones suaves.
    """
    if len(images) == 0:
        raise ValueError("Se requiere al menos una imagen.")
    if len(images) == 1:
        return images[0].copy()
    if len(homographies) != len(images) - 1:
        raise ValueError("Se esperan n-1 homografías que mapeen img_i -> img_0.")
    if blend.lower() != "feather":
        raise ValueError("Por ahora solo se soporta blend='feather'.")

    num_images = len(images)

    # Homografías al sistema de la imagen 0
    H_to_ref: List[np.ndarray] = [np.eye(3, dtype=np.float32)]
    for H in homographies:
        H_to_ref.append(H.astype(np.float32))

    # Calcular el lienzo que contiene todas las imágenes transformadas
    all_corners = []
    for idx, (img, H) in enumerate(zip(images, H_to_ref)):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        warped_corners = cv.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)
    all_corners = np.concatenate(all_corners, axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translate_x, translate_y = -x_min, -y_min
    new_w = int(x_max - x_min)
    new_h = int(y_max - y_min)

    T = np.array([[1, 0, translate_x],
                  [0, 1, translate_y],
                  [0, 0, 1]], dtype=np.float32)

    # Preparar acumuladores
    accumulator = np.zeros((new_h, new_w, 3), dtype=np.float32)
    dist_sums = np.zeros((new_h, new_w), dtype=np.float32)

    # Warpear imágenes y construir pesos por distancia (feathering)
    for img, H in zip(images, H_to_ref):
        H_canvas = T @ H
        warped = cv.warpPerspective(img, H_canvas, (new_w, new_h))

        # Máscara binaria de cobertura de la imagen warpeada
        h_i, w_i = img.shape[:2]
        src_mask = np.ones((h_i, w_i), dtype=np.uint8) * 255
        mask = cv.warpPerspective(src_mask, H_canvas, (new_w, new_h))

        # Mapa de distancias dentro de la máscara (0 fuera, >0 dentro)
        dist = cv.distanceTransform(mask, distanceType=cv.DIST_L2, maskSize=5)

        dist_sums += dist
        # Acumular imagen ponderada por distancia (se normaliza después)
        accumulator += warped.astype(np.float32) * dist[..., None]

    # Normalización y conversión
    eps = 1e-6
    dist_sums_safe = dist_sums + eps
    panorama = accumulator / dist_sums_safe[..., None]
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    return panorama