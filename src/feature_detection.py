from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import Tuple, Literal

DetectorName = Literal["SIFT", "ORB", "AKAZE"]

def detect_and_describe(image: np.ndarray, method: DetectorName = "SIFT") -> Tuple[list[cv.KeyPoint], np.ndarray]:
    """Detecta keypoints y computa descriptores.

    Parameters
    ----------
    image : np.ndarray
        Imagen BGR o GRAY.
    method : {"SIFT","ORB","AKAZE"}
        Detector/descriptor a utilizar.

    Returns
    -------
    (kps, desc): Tuple[list[cv.KeyPoint], np.ndarray]
        Lista de keypoints y matriz (N x D) de descriptores.
    """
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if method == "SIFT":
        extractor = cv.SIFT_create()  # requiere opencv-contrib
    elif method == "ORB":
        extractor = cv.ORB_create(nfeatures=3000, scoreType=cv.ORB_HARRIS_SCORE)
    elif method == "AKAZE":
        extractor = cv.AKAZE_create()
    else:
        raise ValueError(f"Unknown method: {method}")

    kps, desc = extractor.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        raise RuntimeError("No se encontraron características; ajusta parámetros o detector.")
    return kps, desc
