from typing import Any

from mlmc.xp import xp

Generator = xp.random.Generator

class EulerMaruyama:
    """ Numerical integrator for SDEs.
        
        Each call to "step" computes a step of the Euler-Maruyama integration
        starting from the given initial conditions x_t.
        Assumes time invariant processes with additive noise.

        $$
        x_{t+1} = x_t + F(x_t)*dt + sqrt(dt) * S * dW_t
        $$

        Arguments:
        
        - h                 : dt, the desired time step
        - E                 : x -> F(x) the force
        - S                 : standard deviation of the noise
    """
    def __init__(self, h  : float,
                       F : Any, # the derivative of the potential
                       S : xp.ndarray # the Standard deviation of the noise
                       ) -> None:
        assert h > 0.0
        self.h = h
        self.sqrthS = xp.sqrt(h)*S.transpose()
        self.F = F
        
    def step(self, x0 : xp.ndarray,
                   gen : Generator) -> xp.ndarray:
        F = self.F(x0)

        Z = gen.standard_normal( x0.shape )
        dx = F*self.h + Z@self.sqrthS
        
        return x0 + dx
