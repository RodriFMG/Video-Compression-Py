import cv2
import numpy as np
from typing import List, Tuple
from Encoding import Encoding
from Utils.block_utils import PointMotion, set_block_ref
from IFrame import IFrame


class PFrame(Encoding):
    def __init__(self, reference: IFrame, residual: np.ndarray,
                 motion_vectors: List[PointMotion]):
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
