from abc import ABC, abstractmethod

from typing import Tuple
from mlmc.xp import xp

class MLMC_l(ABC):

    def compute(self, N:int) -> Tuple[ xp.ndarray, float ]:
        """ Returns the vector containing the estimations AND the cost.
        """
        pass

class MLMC_l_AbstractFactory(ABC):

    @abstractmethod
    def create(self, l:int) -> MLMC_l:
        """ Create an instance of an object that computes the estimators for
        level l. The concrete class MAY throw an error for l >= L_max, for some
        L_max
        """
        pass
