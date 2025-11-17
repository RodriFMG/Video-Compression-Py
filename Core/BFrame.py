import cv2
import numpy as np
from typing import List, Optional
from Encoding import Encoding
from Utils.block_utils import PointMotion, interpolation, set_block_ref
from Utils.motion_utils import motion_block, residual_bframe
from IFrame import IFrame
from PFrame import PFrame


def apply_motions(frame: Encoding):
    if isinstance(frame, BFrame):
        for motion in frame.MVprev:
            frame.set_block(frame.prevRef, motion)
        for motion in frame.MVnext:
            frame.set_block(frame.nextRef, motion)
    else:
        for motion in frame.MV:
            frame.ref.set_block(motion)


# declaracion circular!
class BFrame(Encoding):
    def __init__(self):
        self.prevRef: Optional[Encoding] = None
        self.nextRef: Optional[Encoding] = None
        self.MVprev: Optional[list] = None
        self.MVnext: Optional[list] = None
        self.res: Optional[np.ndarray] = None
        self.block_size: Optional[int] = None
        self.alpha: Optional[float] = None

    def set_block(self, ref: Encoding, pm: PointMotion):
        set_block_ref(ref.decode(), pm, self.block_size)

    def decode(self) -> np.ndarray:
        # Aplicar motion vectors a referencias

        apply_motions(self.prevRef)
        apply_motions(self.nextRef)

        ref_decoded_prev = self.prevRef.decode()
        ref_decoded_next = self.nextRef.decode()
        interpolated = interpolation(ref_decoded_prev, ref_decoded_next)
        result = cv2.add(interpolated, self.res)
        return result

    def encode(self, prev_frame: Encoding, frame: np.ndarray, fut_frame: Encoding, block_size: int,
               alpha: float = 0.5) -> "BFrame":

        # Decodificacion del frame previo y futuro
        prev_frame_dec = prev_frame.decode()
        fut_frame_dec = fut_frame.decode()

        height, width = frame.shape[:2]
        mv_prev = []
        mv_fut = []

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                mv_prev.append(motion_block(prev_frame_dec, frame, i, j, block_size))

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                mv_fut.append(motion_block(fut_frame_dec, frame, i, j, block_size))

        residual = residual_bframe(mv_prev, mv_fut, prev_frame_dec,
                                   frame, fut_frame_dec, block_size, alpha)

        prev_ref = prev_frame
        next_ref = fut_frame

        self.prevRef = prev_ref
        self.nextRef = next_ref
        self.MVprev = mv_prev
        self.MVnext = mv_fut
        self.res = residual.copy()
        self.block_size = block_size
        self.alpha = alpha

        return self
