import numpy as np
from typing import Optional
from Encoding import Encoding


class IFrame(Encoding):
    def __init__(self):
        self.ref: Optional[np.ndarray] = None
        self.block_size: Optional[int] = None

    def decode(self) -> np.ndarray:
        return self.ref

    def encode(self, frame: np.ndarray, block_size: int) -> "IFrame":
        self.ref = frame.copy()
        self.block_size = block_size
        return self
