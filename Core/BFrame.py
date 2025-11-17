import cv2
import numpy as np
from typing import List
from Encoding import Encoding
from Utils.block_utils import PointMotion, interpolation
from IFrame import IFrame
from PFrame import PFrame

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