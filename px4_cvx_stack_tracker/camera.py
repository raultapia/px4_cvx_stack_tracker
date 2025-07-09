import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float

    def backproject(self, u: float, v: float, d: float, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
        x_c = (u - self.cx) / self.fx * d
        y_c = (v - self.cy) / self.fy * d
        z_c = d
        p_c = np.array([x_c, y_c, z_c], dtype=float)
        return R_wc @ p_c + t_wc
