import cv2
import numpy as np
from typing import List, Tuple
from Encoding import Encoding
from Utils.block_utils import PointMotion, set_block_ref


class IFrame(Encoding):
    def __init__(self, reference: np.ndarray, block_size: int):
        self.ref = reference.copy()
        self.block_size = block_size

    def set_block(self, pm: PointMotion):
        set_block_ref(self.ref, pm, self.block_size)

    def decode(self) -> np.ndarray:
        return self.ref
