from src.systems.base import BaseSystem
import numpy as np

class Pendulum(BaseSystem):
    def __init__(self, name: str, attractor_radius: float = 0.1):
        super().__init__(name)
        self.attractor_radius = attractor_radius
        
    def attractors(self):
        return [
            [0, 0],
            [2.1, 0],
            [-2.1, 0],
        ]
        
    def is_in_attractor(self, x: np.ndarray) -> bool:
        for attractor in self.attractors():
            if np.linalg.norm(x - attractor) < self.attractor_radius:
                return True
        return False
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        data[:, 0] = data[:, 0] / 3.14
        data[:, 1] = data[:, 1] / (2.0 * 3.14)
        return data
    
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        data[:, 0] = data[:, 0] * 3.14
        data[:, 1] = data[:, 1] * (2.0 * 3.14)
        return data