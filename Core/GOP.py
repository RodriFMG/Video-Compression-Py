from typing import List
import numpy as np
from Encoding import Encoding


class GOP:
    def __init__(self, size: int):
        self.group: List[Encoding] = []
        self.size = size

    def add_frame(self, frame_enc: Encoding):
        if len(self.group) == self.size:
            raise RuntimeError("ERROR: Cantidad máxima del grupo alcanzada...")
        self.group.append(frame_enc)

    def decode(self) -> List[np.ndarray]:
        group_decode = []
        for frame_enc in self.group:
            group_decode.append(frame_enc.decode())
        return group_decode

    def is_full(self) -> bool:
        return len(self.group) == self.size
