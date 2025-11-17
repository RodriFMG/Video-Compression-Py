import numpy as np
from typing import List

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
