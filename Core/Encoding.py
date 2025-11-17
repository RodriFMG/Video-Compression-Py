from abc import ABC, abstractmethod
import numpy as np


class Encoding(ABC):

    @abstractmethod
    def decode(self) -> np.ndarray:
        raise NotImplementedError
