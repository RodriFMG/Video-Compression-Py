from typing import List
import numpy as np
from Encoding import Encoding
from BFrame import BFrame
from PFrame import PFrame
from IFrame import IFrame


class GOP:
    def __init__(self):
        # Acá defino la secuencia de frames que seguirá...
        self.group: List[Encoding] = [IFrame(),
                                      BFrame(), BFrame(), PFrame(),
                                      BFrame(), BFrame(), PFrame(),
                                      BFrame(), PFrame()]
        self.size: int = len(self.group)

    # Decode basado en PTS <- Presentation Time Stamp
    def decode(self) -> List[np.ndarray]:
        group_decode = []
        for frame_enc in self.group:
            group_decode.append(frame_enc.decode())
        return group_decode

    # Encode basado en DST <- Decoding/Encoding Time Stamp
    def encode(self, frames: list[np.ndarray], block_size: int, alpha: float):
        past_idx_encode = 0
        n = min(self.size, len(frames))
        self.size = n

        for idx in range(n):

            # Si el último frame es BFrame se cambia a PFrame para poder calcularlo...
            if idx+1 == n and isinstance(self.group[idx], BFrame):
                self.group[idx] = PFrame()

            # Acá en el PFrame en caso sea una secuencia de I B B B B P B B P, ya estoy codificando
            # el Pframe con el Iframe
            if isinstance(self.group[idx], PFrame):
                self.group[idx].encode(self.group[past_idx_encode], frames[idx], block_size)
            elif isinstance(self.group[idx], IFrame):
                self.group[idx].encode(frames[idx], block_size)

            if not isinstance(self.group[idx], BFrame):

                # +1 para que empiece desde BFrame
                for step in range(past_idx_encode+1, idx):
                    prev_frame: Encoding = self.group[past_idx_encode]
                    fut_frame: Encoding = self.group[idx]
                    self.group[step].encode(prev_frame, frames[step], fut_frame, block_size, alpha=alpha)

                past_idx_encode = idx

    def get_size_group(self):
        return self.size
