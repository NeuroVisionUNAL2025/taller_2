from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import Tuple, List

def estimate_homography(kp1, kp2, matches, ransac_thresh: float = 3.0, confidence: float = 0.995) -> Tuple[np.ndarray, np.ndarray]:
    """Estima H con RANSAC a partir de keypoints y matches.

    Returns
    -------
    H : (3,3) homografía
    mask : inliers (Nx1)
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, ransacReprojThreshold=ransac_thresh, confidence=confidence)
    if H is None:
        raise RuntimeError("No se pudo estimar homografía; revisa correspondencias o parámetros.")
    return H, mask

def warp_images(base: np.ndarray, overlay: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Deforma `overlay` al sistema de `base` usando H y retorna (mosaico, mascara_overlay)."""
    h1, w1 = base.shape[:2]
    h2, w2 = overlay.shape[:2]

    corners = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    warped_corners = cv.perspectiveTransform(corners, H)
    all_corners = np.concatenate((warped_corners, np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    tx, ty = -xmin, -ymin

    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
    mosaic = cv.warpPerspective(overlay, T @ H, (xmax-xmin, ymax-ymin))
    mosaic[ty:ty+h1, tx:tx+w1] = np.where(mosaic[ty:ty+h1, tx:tx+w1]==0, base, mosaic[ty:ty+h1, tx:tx+w1])
    mask = np.zeros((ymax-ymin, xmax-xmin), dtype=np.uint8)
    mask[ty:ty+h1, tx:tx+w1] = 255
    return mosaic, mask

def feather_blend(img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray) -> np.ndarray:
    """Blending por feathering usando distancia al borde."""
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (np.any(img2!=0, axis=2)).astype(np.uint8)

    overlap = cv.bitwise_and(m1, m2)
    only1 = cv.bitwise_and(m1, cv.bitwise_not(m2))
    only2 = cv.bitwise_and(m2, cv.bitwise_not(m1))

    result = img2.copy()
    result[only1.astype(bool)] = img1[only1.astype(bool)]

    if np.any(overlap):
        dist1 = cv.distanceTransform((m1*255).astype(np.uint8), cv.DIST_L2, 5)
        dist2 = cv.distanceTransform((m2*255).astype(np.uint8), cv.DIST_L2, 5)
        w1 = dist1 / (dist1 + dist2 + 1e-6)
        w2 = 1.0 - w1
        w1 = np.expand_dims(w1, axis=2)
        w2 = np.expand_dims(w2, axis=2)
        mixed = (img1.astype(np.float32)*w1 + img2.astype(np.float32)*w2).astype(np.uint8)
        result[overlap.astype(bool)] = mixed[overlap.astype(bool)]
    return result

def stitch_sequence(images: List[np.ndarray], homography_pairs: List[Tuple[int,int,np.ndarray]]) -> np.ndarray:
    """Construye un mosaico a partir de una lista de imágenes y homografías entre pares.
    homography_pairs: lista de (idx_src, idx_dst, H_src_to_dst)
    """
    canvas = images[0].copy()
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    mask[:,:] = 255

    for (src_idx, dst_idx, H) in homography_pairs:
        if dst_idx != 0:
            raise NotImplementedError("Implementación básica: las H deben mapear al índice 0 (referencia).")
        warped, mask_src = warp_images(images[src_idx], images[dst_idx], H)
        canvas = feather_blend(warped, canvas, mask_src)
    return canvas
