import cv2
import numpy as np
from typing import List, Tuple


# ---------------------------
# Estructura de Motion Vector
# ---------------------------
class PointMotion:
    def __init__(self, pos: Tuple[int, int], dx: int, dy: int):
        self.pos = pos  # (x, y)
        self.dx = dx
        self.dy = dy


# ---------------------------
# Funciones auxiliares
# ---------------------------
def set_block_ref(ref: np.ndarray, pm: PointMotion, block_size: int):
    x, y = pm.pos
    src_block = ref[y:y + block_size, x:x + block_size].copy()
    ref[y + pm.dy:y + pm.dy + block_size, x + pm.dx:x + pm.dx + block_size] = src_block


def interpolation(prev: np.ndarray, next: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(prev, 1.0 - alpha, next, alpha, 0.0)


# ---------------------------
# Clases Base
# ---------------------------
class Encoding:
    def decode(self) -> np.ndarray:
        raise NotImplementedError


# ---------------------------
# I-Frame
# ---------------------------
class IFrame(Encoding):
    def __init__(self, reference: np.ndarray, block_size: int):
        self.ref = reference.copy()
        self.block_size = block_size

    def set_block(self, pm: PointMotion):
        set_block_ref(self.ref, pm, self.block_size)

    def decode(self) -> np.ndarray:
        return self.ref


# ---------------------------
# P-Frame
# ---------------------------
class PFrame(Encoding):
    def __init__(self, reference: IFrame, residual: np.ndarray, motion_vectors: List[PointMotion]):
        self.ref = reference
        self.res = residual.copy()
        self.MV = motion_vectors

    def set_block(self, pm: PointMotion):
        self.ref.set_block(pm)

    def decode(self) -> np.ndarray:
        # Aplicar motion vectors
        for motion in self.MV:
            self.ref.set_block(motion)
        ref_decoded = self.ref.decode()
        result = cv2.add(ref_decoded, self.res)
        return result


# ---------------------------
# B-Frame
# ---------------------------
class BFrame(Encoding):
    def __init__(self, prev_ref: IFrame, next_ref: PFrame,
                 motion_vectors_prev: List[PointMotion],
                 motion_vectors_next: List[PointMotion],
                 residual: np.ndarray):
        self.prevRef = prev_ref
        self.nextRef = next_ref
        self.MVprev = motion_vectors_prev
        self.MVnext = motion_vectors_next
        self.res = residual.copy()

    def decode(self) -> np.ndarray:
        # Aplicar motion vectors a referencias
        for motion in self.MVprev:
            self.prevRef.set_block(motion)
        for motion in self.MVnext:
            self.nextRef.set_block(motion)

        ref_decoded_prev = self.prevRef.decode()
        ref_decoded_next = self.nextRef.decode()
        interpolated = interpolation(ref_decoded_prev, ref_decoded_next)
        result = cv2.add(interpolated, self.res)
        return result
