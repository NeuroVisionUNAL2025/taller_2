from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import Sequence, Tuple, Literal

MatchStrategy = Literal["BF", "FLANN"]

def match_descriptors(desc1: np.ndarray, desc2: np.ndarray, strategy: MatchStrategy = "BF", ratio: float = 0.75) -> list[cv.DMatch]:
    """Empareja descriptores usando KNN y filtra con la regla de Lowe.

    - Para SIFT/AKAZE (float): usa L2
    - Para ORB (binario): usa Hamming
    """
    is_binary = desc1.dtype == np.uint8
    if strategy == "BF":
        norm = cv.NORM_HAMMING if is_binary else cv.NORM_L2
        matcher = cv.BFMatcher(norm)
    else:
        if is_binary:
            # FLANN para binario no es estable; mejor BF. Aun así:
            index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                table_number=12, key_size=20, multi_probe_level=2)
            search_params = dict(checks=64)
            matcher = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        else:
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=64)
            matcher = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    good.sort(key=lambda m: m.distance)
    return good

def keypoints_to_points(kps: Sequence[cv.KeyPoint], matches: Sequence[cv.DMatch], side: Literal["query","train"]) -> np.ndarray:
    """Convierte matches a un arreglo Nx2 de puntos (x,y)."""
    if side == "query":
        pts = np.float32([kps[m.queryIdx].pt for m in matches])
    else:
        pts = np.float32([kps[m.trainIdx].pt for m in matches])
    return pts
