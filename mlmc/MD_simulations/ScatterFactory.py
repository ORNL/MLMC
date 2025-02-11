from math import pi, sin, cos

from mlmc.scatter.scatter import Scatter
from mlmc.xp import xp

from typing import Tuple

def ScatterFactory(Rc:float, D:float, theta:float) -> Tuple[Scatter, xp.ndarray] :
        assert Rc < D
        theta *= pi/180.0 # lattice angle (60 = hexagonal)
        L = xp.array([[D, 0.0],
                  [cos(theta)*D, sin(theta)*D]
                 ])
        x_0 = L.sum(0)/3
        return Scatter(L, Rc), x_0
