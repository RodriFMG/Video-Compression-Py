import cv2
import numpy as np
from typing import List, Tuple, Optional
from Encoding import Encoding
from Utils.block_utils import PointMotion, set_block_ref


class IFrame(Encoding):
    def __init__(self):
        self.ref: Optional[np.ndarray] = None
        self.block_size: Optional[int] = None

    def set_block(self, pm: PointMotion):
        set_block_ref(self.ref, pm, self.block_size)

    def decode(self) -> np.ndarray:
        return self.ref

    def encode(self, frame: np.ndarray, block_size: int) -> "IFrame":
        self.ref = frame.copy()
        self.block_size = block_size
        return self
