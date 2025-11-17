from abc import ABC, abstractmethod
import numpy as np
from Utils.block_utils import PointMotion


class Encoding(ABC):

    @abstractmethod
    def decode(self) -> np.ndarray:
        raise NotImplementedError
