import numpy as np
from mlmc.xp import xp

from mlmc.MLMC.MLMC_L import MLMC_l_AbstractFactory, MLMC_l
from mlmc.Integration_SDE.EulerMaruyama import EulerMaruyama

class MLMC_l_Geometric(MLMC_l):

    def __init__(self, l:int, M:int, x0:xp.ndarray, dt0:float, T:float, F:any, G:xp.ndarray, phi: any):
        """ l:      the level estimator that the object will return
                        E.g. 0 return P0, 1 returns P1 - P0, l returns Pl - P(l-1)
            M:      the number of steps of level l+1 for every step of level l+1
            x0:     the initial conditions for the problem
            dt0:    the stepsize of level 0
            T:      the time horizon (assume dt0*integer...)
            F:      the function F(x)
            G:      the covariance of the additive noise
            phi:    the function to estimate
        """
        super().__init__()
        self.l = l
        if(l==0):
            self.compute = self._integrate
        else:
            self.compute = self._difference

        self._x0 = xp.copy(x0)
        self._n = len(self._x0)
        self._dt0 = dt0
        self._T = T
        self._phi = phi
        self._steps0 = int(np.floor(T/dt0))
        self._F = F
        self._G = G
        self._M = M
        self._gen = xp.random.default_rng(seed=None) # WARNING: cupy is slow, always use numpy and move it. It is faster.

    def _integrate(self, N):
        # Initialize the integrator
        x = xp.zeros( (N,self._n) )
        x[:,:] = xp.copy(self._x0)[None,:]

        sqrthS = xp.sqrt(self._dt0)*self._G.transpose()
        for i in xp.arange(self._steps0):
            Z = self._gen.standard_normal( x.shape )
            dx = self._F(x)*self._dt0 + Z@sqrthS
            x += dx

        return self._phi(x), N*self._steps0

    def _difference(self, N):
        dt = self._dt0/(self._M**self.l) # Fine
        dtc = dt*self._M # Coarse

        k = int(xp.floor(self._T/dt))
        k0 = int(xp.floor(k/self._M))
        sqrtSf = np.sqrt(dt)*self._G.transpose()
        xf = xp.zeros( (N,self._n) )
        xf[:,:] = xp.copy(self._x0)[None,:]
        xc = xp.copy(xf)
        
        for i in range(k0):
            dWc = xp.zeros( xc.shape )

            # Fine steps (intermediate...)
            for j in range(self._M):
                Z = self._gen.standard_normal( xf.shape )
                dWf = Z@sqrtSf
                
                F = self._F(xf)
                
                dxf = F*dt + dWf
                xf += dxf
                dWc += dWf

            F = self._F(xc)
            dxc = F*dtc + dWc
            xc += dxc

        Pl = self._phi(xf) - self._phi(xc)

        return Pl, N*(k + k0)
