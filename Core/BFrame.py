import numpy as np
from typing import Optional
from Encoding import Encoding
from Utils.block_utils import interpolation, set_block_ref
from Utils.motion_utils import motion_block, residual_bframe


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

    def decode(self) -> np.ndarray:

        # Frames decodificados de referencia, copias para aplicar el motion.
        # Esto es una "recursiva" que retrocede hasta el IFrame...?
        pred_prev = self.prevRef.decode().copy()
        pred_fut = self.nextRef.decode().copy()

        # Aplicando los vectores de movimiento a los frames decodificados
        for mv in self.MVprev:
            set_block_ref(pred_prev, mv, self.block_size)

        for mv in self.MVnext:
            set_block_ref(pred_fut, mv, self.block_size)

        # Predicción del BFrame
        prediction = interpolation(pred_prev, pred_fut, alpha=self.alpha)

        # Decodificación + llevarlo a [0, 255]
        decoded = self.res.astype(np.int16) + prediction.astype(np.int16)
        decoded = np.clip(decoded, 0, 255).astype(np.uint8)

        return decoded

    def encode(self, prev_frame: Encoding, frame: np.ndarray, fut_frame: Encoding, block_size: int,
               alpha: float = 0.5) -> "BFrame":

        # Decodificacion del frame previo y futuro
        prev_frame_dec = prev_frame.decode().copy()
        fut_frame_dec = fut_frame.decode().copy()

        height, width = frame.shape[:2]
        mv_prev = []
        mv_fut = []

        # Calculando vectores de movimiento
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                mv_prev.append(motion_block(prev_frame_dec, frame, i, j, block_size))

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                mv_fut.append(motion_block(fut_frame_dec, frame, i, j, block_size))

        # Predicción
        residual = residual_bframe(mv_prev, mv_fut, prev_frame_dec,
                                   frame, fut_frame_dec, block_size, alpha)

        # Configuración del frame
        self.prevRef = prev_frame
        self.nextRef = fut_frame
        self.MVprev = mv_prev
        self.MVnext = mv_fut
        self.res = residual.copy()
        self.block_size = block_size
        self.alpha = alpha

        return self
