import cv2
import numpy as np
from typing import List, Tuple
from Encoding import Encoding
from Utils.block_utils import PointMotion, set_block_ref
from IFrame import IFrame
from BFrame import BFrame
from Utils.motion_utils import motion_block, residual_pframe
from typing import Optional


def apply_motions(frame: Encoding):
    if isinstance(frame, BFrame):
        for motion in frame.MVprev:
            frame.set_block(frame.prevRef, motion)
        for motion in frame.MVnext:
            frame.set_block(frame.nextRef, motion)
    else:
        for motion in frame.MV:
            frame.ref.set_block(motion)


class PFrame(Encoding):
    def __init__(self):
        self.ref: Optional[Encoding] = None
        self.res: Optional[np.ndarray] = None
        self.MV: Optional[list] = None
        self.block_size: Optional[int] = None

    def set_block(self, pm: PointMotion):
        set_block_ref(self.ref.decode(), pm, self.block_size)

    def decode(self) -> np.ndarray:
        # Aplicar motion vectors

        apply_motions(self.ref)

        ref_decoded = self.ref.decode()
        result = cv2.add(ref_decoded, self.res)
        return result

    def encode(self, prev_frame: Encoding, frame: np.ndarray, block_size: int) -> "PFrame":

        # Decodificación del frame previo
        prev_dec = prev_frame.decode()

        height, width = frame.shape[:2]
        motion_vectors = []

        # Calculando vectores de movimiento
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                motion_vectors.append(motion_block(prev_dec, frame, i, j, block_size))

        # Predicción
        residual = residual_pframe(motion_vectors, prev_dec, frame, block_size)

        self.ref = prev_frame
        self.res = residual
        self.MV = motion_vectors
        self.block_size = block_size

        return self
