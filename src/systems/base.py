from abc import ABC, abstractmethod
import numpy as np

class BaseSystem(ABC):
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        pass
    