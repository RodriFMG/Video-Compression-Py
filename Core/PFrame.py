import numpy as np
from Encoding import Encoding
from Utils.block_utils import set_block_ref
from Utils.motion_utils import motion_block, residual_pframe
from typing import Optional


class PFrame(Encoding):
    def __init__(self):
        self.ref: Optional[Encoding] = None
        self.res: Optional[np.ndarray] = None
        self.MV: Optional[list] = None
        self.block_size: Optional[int] = None

    def decode(self) -> np.ndarray:

        # Frame decodificado de referencia, copias para aplicar el motion.
        pred_ref = self.ref.decode().copy()

        # Aplicando los vectores de movimiento a los frames decodificados
        for mv in self.MV:
            set_block_ref(pred_ref, mv, self.block_size)

        # La predicción vendría siendo el frame decodificado aplicando los motion vectors
        decoded = self.res.astype(np.int16) + pred_ref.astype(np.int16)
        decoded = np.clip(decoded, 0, 255).astype(np.uint8)

        return decoded

    def encode(self, prev_frame: Encoding, frame: np.ndarray, block_size: int) -> "PFrame":

        # Decodificación del frame previo
        prev_dec = prev_frame.decode().copy()

        height, width = frame.shape[:2]
        motion_vectors = []

        # Calculando vectores de movimiento
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                motion_vectors.append(motion_block(prev_dec, frame, i, j, block_size))

        # Predicción
        residual = residual_pframe(motion_vectors, prev_dec, frame, block_size)

        # Configuración del frame
        self.ref = prev_frame
        self.res = residual
        self.MV = motion_vectors
        self.block_size = block_size

        return self
