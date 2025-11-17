import numpy as np
from typing import List

from Core.IFrame import IFrame
from Core.PFrame import PFrame
from Core.BFrame import BFrame

from Utils.block_utils import interpolation, PointMotion, set_block_ref


# ---------------------------
# Patrón Diamond Search (simplificado)
# ---------------------------
DiamondSearch = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)]


# ---------------------------
# Función MotionBlock
# ---------------------------
def motion_block(prev_frame: np.ndarray, frame: np.ndarray, i: int, j: int, block_size: int) -> PointMotion:
    height, width = prev_frame.shape[:2]
    best_dx, best_dy = 0, 0
    min_sad = float('inf')

    for dx, dy in DiamondSearch:
        ref_x = j + dx * block_size
        ref_y = i + dy * block_size

        if ref_x < 0 or ref_y < 0 or ref_x + block_size > width or ref_y + block_size > height:
            continue

        sad = np.sum(np.abs(frame[i:i + block_size, j:j + block_size].astype(np.int32) -
                            prev_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size].astype(np.int32)))

        if sad < min_sad:
            min_sad = sad
            best_dx, best_dy = dx, dy

    return PointMotion((i, j), best_dx, best_dy)


# ---------------------------
# Función residual para P-frame
# ---------------------------
def residual_pframe(motion_vectors: List[PointMotion], prev_frame: np.ndarray, frame: np.ndarray,
                    block_size: int) -> np.ndarray:
    predict_frame = prev_frame.copy()
    for motion in motion_vectors:
        set_block_ref(predict_frame, motion, block_size)
    residual = frame.astype(np.int16) - predict_frame.astype(np.int16)
    return residual


# ---------------------------
# Función residual para B-frame
# ---------------------------
def residual_bframe(mv_prev: List[PointMotion], mv_fut: List[PointMotion],
                    prev_frame: np.ndarray, frame: np.ndarray, fut_frame: np.ndarray,
                    block_size: int, alpha: float = 0.5) -> np.ndarray:
    pred_prev = prev_frame.copy()
    pred_fut = fut_frame.copy()

    for motion in mv_prev:
        set_block_ref(pred_prev, motion, block_size)
    for motion in mv_fut:
        set_block_ref(pred_fut, motion, block_size)

    predict_frame = interpolation(pred_prev, pred_fut, alpha)
    residual = frame.astype(np.int16) - predict_frame.astype(np.int16)
    return residual


# ---------------------------
# Computar I, P y B frames
# ---------------------------
def compute_iframe(frame: np.ndarray, block_size: int) -> IFrame:
    return IFrame(frame, block_size)


def compute_pframe(prev_frame: np.ndarray, frame: np.ndarray, block_size: int) -> PFrame:
    height, width = prev_frame.shape[:2]
    motion_vectors = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            motion_vectors.append(motion_block(prev_frame, frame, i, j, block_size))
    residual = residual_pframe(motion_vectors, prev_frame, frame, block_size)
    ref_iframe = IFrame(prev_frame, block_size)
    return PFrame(ref_iframe, residual, motion_vectors)


def compute_bframe(prev_frame: np.ndarray, frame: np.ndarray, fut_frame: np.ndarray, block_size: int,
                   alpha: float = 0.5) -> BFrame:
    height, width = prev_frame.shape[:2]
    mv_prev = []
    mv_fut = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            mv_prev.append(motion_block(prev_frame, frame, i, j, block_size))

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            mv_fut.append(motion_block(fut_frame, frame, i, j, block_size))

    residual = residual_bframe(mv_prev, mv_fut, prev_frame, frame, fut_frame, block_size, alpha)
    prev_ref = IFrame(prev_frame, block_size)
    next_ref = PFrame(IFrame(fut_frame, block_size), np.zeros_like(residual), mv_fut)
    return BFrame(prev_ref, next_ref, mv_prev, mv_fut, residual)
