from mlmc.MLMC.Factory import IntegratorFactory
from mlmc.Integration_SDE.EulerMaruyama import EulerMaruyama

from mlmc.xp import xp

class EMFactory(IntegratorFactory):
    """ Immutable objects: all the scatter grids produced by this class use L, Rc
    given at initialization time
    """
    def __init__(self, beta:float, sigma:float, S) -> None:
        """ beta and sigma are the parameters of the integration
        S must have the moethod S.dE(x), the gradient of the potential
        computed at x.
        The class creates gamma, F automatically.
        """
        assert beta > 0
        assert sigma > 0
        super()
        self.sigma = sigma
        self.gamma = 0.5*beta*(sigma**2)
        self.S = S

    def F(self,x):
        return -self.gamma*self.S.dE(x)

    def getIntegrator(self,h):
        """ Warning, returns a different integrator for every call.
        They might differ for the parameter h, the integration step.
        """
        return EulerMaruyama(h, self.F, self.sigma*xp.eye(2))
