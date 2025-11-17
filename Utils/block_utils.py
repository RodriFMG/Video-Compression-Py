import cv2
import numpy as np
from typing import List, Tuple


class PointMotion:
    def __init__(self, pos: Tuple[int, int], dx: int, dy: int):
        self.pos = pos  # (x, y)
        self.dx = dx
        self.dy = dy


def set_block_ref(ref: np.ndarray, pm: PointMotion, block_size: int):
    x, y = pm.pos
    src_block = ref[y:y + block_size, x:x + block_size].copy()
    ref[y + pm.dy:y + pm.dy + block_size, x + pm.dx:x + pm.dx + block_size] = src_block


def interpolation(prev: np.ndarray, next: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(prev, 1.0 - alpha, next, alpha, 0.0)
