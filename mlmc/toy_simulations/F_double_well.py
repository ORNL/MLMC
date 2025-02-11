from mlmc.xp import xp

class F_double_well:
    def __init__(self, mu:float, beta:float):
        self._mu = mu
        self._beta = beta
        self._mu2 = mu*mu

    def E(self, x:xp.ndarray):
        """ Returns the potential
        """
        x2 = x*x
        x4 = x2*x2
        part = 0.25*x4 - 0.5*self._mu2*x2
        return self._beta*part
    
    def dE(self, x:xp.ndarray):
        """ Returns the force=-potential derivative
        """
        x_pm = x + self._mu             # 1 operation
        x_mm = x - self._mu             # 1 operation
        return -self._beta*x*x_pm*x_mm  # 4 operations 
        # total is 6 operations
    
    def d2E(self, x:xp.ndarray):
        """ Returns the derivative of the force
        """
        x2 = x*x                        # 1 operation
        part = 3*x2 - self._mu2         # 2 operations
        return -self._beta*part         # 2 operations
        # total is 5 operations