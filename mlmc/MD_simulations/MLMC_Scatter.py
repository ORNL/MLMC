from typing import Optional

import numpy as np
from mlmc.xp import xp

from mlmc.MLMC.MLMC_L import MLMC_l_AbstractFactory, MLMC_l

from mlmc.MLMC.MLMC_L_Geometric import MLMC_l_Geometric

from .ScatterFactory import Scatter
from mlmc.Integration_SDE.EulerMaruyama import EulerMaruyama

from enum import Enum
class MLMC_Constraint(str, Enum):
    GEOM = "Geom"
    SPRING_AN = "Spring"
    SPRING_AN_CAP = "Spring_Cap"
    SPRING_FIXED = "Spring_F"

class MLMC_l_Scatter(MLMC_l):
    _eps = 1e-6
    def __init__(self, l:int, x0:xp.ndarray, dt0:float, T:float, s:Scatter, beta:float, sigma:float, phi: any, cap:float=-1, fixed:bool=False):
        """ l:      the level estimator that the object will return
                        E.g. 0 return P0, 1 returns P1 - P0, l returns Pl - P(l-1)
            x0:     the initial conditions for the problem
            dt0:    the stepsize of level 0
            T:      the time horizon (assume dt0*integer...)
            phi:    the function to estimate
        """
        super().__init__()
        self.l = l
        if(l==0):
            self.compute = self._integrate
        else:
            self.compute = self._spring

        self.x0 = x0
        self.n = len(self.x0)
        self.dt0 = dt0
        self.T = T
        self.phi = phi
        self.steps0 = int(np.floor(T/dt0))
        self.sigma = sigma*xp.eye(self.n)
        self._sigma_diag = xp.diag(self.sigma)
        self._gamma = 0.5*beta*(sigma**2)
        self.gen = xp.random.default_rng(seed=None)
        self.S = s

        if fixed:
            self._K = self._K_fixed
            self._cap = cap
            self._cost_s = 0
        else:
            self._K = self._K_analical

            self._cost_s = 2 # It is like adding an operation per iteration, give or take.

            if cap > 0:
                self._K_to_cap = self._K
                self._K = self._K_cap
                self._cap = cap

    def _K_analical(self,dY:xp.ndarray,Yc:xp.ndarray,Yf:xp.ndarray,FYc:xp.ndarray,FYf:xp.ndarray, N:int):
        # If the derivative is known, use it.
        Jf = -self._gamma*self.S.d2E(Yc)

        # Jf*dY
        Jf_dY = Jf*dY[:,:,None] # Jf*dY
        Jf_dY = Jf_dY.sum(1) # Jf*dY
        # dY'*Jf_dY
        dF_dY = Jf_dY*dY # dY'*Jf_dY
        dF_dY = dF_dY.sum(1) # dY'*Jf_dY

        select_1 = dF_dY > 0

        dY_norm_2 = (dY*dY).sum(1)
        select_2 = dY_norm_2 > MLMC_l_Scatter._eps

        select = xp.logical_and(select_1, select_2)
        dF_dY_norm = xp.zeros(N)
        dF_dY_norm[select] = dF_dY[select]/dY_norm_2[select]

        return dF_dY_norm/2

    def _K_fixed(self,dY:xp.ndarray,Yc:xp.ndarray,Yf:xp.ndarray,FYc:xp.ndarray,FYf:xp.ndarray, N:int):
        return self._cap*xp.ones((N,))

    # This is a decorator to use in case we want to cap the Spring stiffness.
    def _K_cap(self,dY:xp.ndarray,Yc:xp.ndarray,Yf:xp.ndarray,FYc:xp.ndarray,FYf:xp.ndarray, N:int):
        S_unbound = self._K_to_cap(dY,Yc,Yf,FYc,FYf, N)
        return xp.minimum(S_unbound,self._cap)

    # This function assigned to the relevant function at initialization....
    # def _K(self,dY:xp.ndarray,Yc:xp.ndarray,Yf:xp.ndarray,FYc:xp.ndarray,FYf:xp.ndarray, N:int):


    def _F(self,x:xp.ndarray):
        return -self._gamma*self.S.dE(x)

    def _integrate(self, N):
        # Initialize the integrator
        em_in_0 = EulerMaruyama(self.dt0, self._F, self.sigma )
        x = xp.zeros( (N,self.n) )
        x[:,:] = xp.copy(self.x0)[None,:]

        for i in np.arange(self.steps0):
            x = em_in_0.step(x,self.gen)

        return self.phi(x), N*self.steps0

    def _spring(self, N):
        dt = self.dt0/(2**self.l) # Fine
        dt1 = dt/2

        k = int(np.floor(self.T/dt))
        k0 = int(np.floor(k/2))

        x0 = xp.zeros( (N,self.n) )
        x0[:,:] = xp.copy(self.x0)[None,:]

        ssqrt1 = self.sigma*np.sqrt(dt)

        # The Radon-Nikodym derivatives
        R_fine_exp = xp.zeros(N)
        R_coarse_exp = xp.zeros(N)

        # NOTE: the spring term is irrelevant at the first step, that's because Yf - Yc = x0 - x0 = 0
        Yf = xp.copy(x0)
        Yc = xp.copy(x0)

        k_cur = xp.zeros( N )
        Kf_hat = xp.zeros( (N, self.n) )

        # Every step of the coarse path is two steps of the fine one
        for j in range(k0):
            # Prepare the two random increments
            dW = ssqrt1*self.gen.standard_normal( (N,self.n,2) )
            dW0 = dW[:,:,0]
            dW1 = dW[:,:,1]

            # Update both the fine path and the coarse path
            Kc_hat = -Kf_hat # Used later in several instructions
            FYc = self._F(Yc)
            Yc_update = ( Kc_hat + FYc )*dt # Yf will be modified

            FYf = self._F(Yf)
            Yc = Yc + Yc_update + dW0
            Yf = Yf + ( Kf_hat + FYf )*dt + dW0 # Yf modified

            # Update the Radon-Nikodym derivative (fine path only)
            R_fine_exp -= self._Radon_Nikodym_cur_exp(dW0, Kf_hat, dt1, N)

            # Update the spring constants
            dY = Yc - Yf
            # k_cur = 1.01*self._K(dY,Yc,Yf,FYc,FYf, N) # inflate the spring a little...

            # Number 2: update only the fine path and complete the update of the coarse path
            Kf_hat = k_cur[:,None]*dY

            FYf = - self._gamma*self.S.dE(Yf)
            Yc = Yc + Yc_update + dW1; # Yc modified.
            Yf = Yf + ( Kf_hat + FYf )*dt + dW1

            # Update the Radon-Nikodym derivative (fine and coarse path)
            R_fine_exp -= self._Radon_Nikodym_cur_exp(dW1, Kf_hat, dt1, N)
            R_coarse_exp -= self._Radon_Nikodym_cur_exp(dW0 + dW1, Kc_hat, dt, N)

            # Update the spring
            dY = Yc - Yf
            k_cur = 1.01*self._K(dY,Yc,Yf,FYc,FYf,N)
            Kf_hat = k_cur[:,None]*dY

        R_fine = xp.exp(R_fine_exp)
        R_coarse = xp.exp(R_coarse_exp)
        Pl = self.phi(Yf)*R_fine - self.phi(Yc)*R_coarse

        return Pl, N*(3+self._cost_s)*k0

    def _Radon_Nikodym_cur_exp(self,dW:xp.ndarray, K_hat:xp.ndarray, dt_half:float, N:int):
        # numpy is SO BAD that you'd rather find another solution to solve in complex cases.
        # K_hat = xp.linalg.solve(self.sigma, K_hat.transpose())
        # dKW = xp.linalg.solve(self.sigma, dW.transpose() + K_hat*dt_half)
        # For now, this class assumes sigma=s*I, so use this...
        K_hat = K_hat/self._sigma_diag
        dKW = (dW + K_hat*dt_half)/self._sigma_diag

        R_cur = K_hat*dKW
        return R_cur.sum(1)

    # This function assigned to the relevant function at initialization....
    # def compute(self, N:int) -> ( np.ndarray, int ):

class MLMC_l_ScatterFactory(MLMC_l_AbstractFactory):

    def __init__(self, beta:float, x0:xp.ndarray, dt0:float, T:float, S:Scatter, phi: any, strategy:MLMC_Constraint=MLMC_Constraint.GEOM, K_max:Optional[float]=None) -> any:
        """ Creates the MLMC_l_Scatter object given the levels l, using the factory method "create"
        Parameters
        ----------
        beta : str
        x0 : str
            The initial position for the simulations
        dt0: float
            The time step at level 0
        T: float
            The final time of the simulations
        S: Scatter
        phi: function
            The function to estimate
        strategy: MLMC_Constraint
            The strategy to use, e.g. Geometrical or adaptive spring. Refer to the enum.
        K_max: floaf
            The maximum spring constant for the given D, given beta=1. If beta is not one, the ode adjust everyhing on its own.
        """
        super().__init__()
        self.beta = beta
        self.dt0 = dt0
        self.x0 = xp.copy(x0)
        self.T = T
        self.phi = phi
        self.S = S
        self.sigma = 1 # Just assume this for now.

        self.gamma = 0.5*beta*(self.sigma**2)
        self.strategy = strategy

        if K_max is not None:
            self.CAP = self.gamma * K_max
            self.K_fix = self.CAP

    def _F(self,x:xp.ndarray):
        return -self.gamma*self.S.dE(x)

    def create(self, l:int) -> MLMC_l_Scatter:
        if self.strategy == MLMC_Constraint.SPRING_FIXED:
            # 5. Version with fixed spring.
            return MLMC_l_Scatter(l, self.x0, self.dt0, self.T, self.S, self.beta, self.sigma, self.phi, cap=self.K_fix, fixed=True)
        if self.strategy == MLMC_Constraint.SPRING_AN_CAP:
            # 4. Version with analitical spring capped.
            return MLMC_l_Scatter(l, self.x0, self.dt0, self.T, self.S, self.beta, self.sigma, self.phi, cap=self.CAP)
        if self.strategy == MLMC_Constraint.SPRING_AN:
            # 3. Version with analitical spring.
            return MLMC_l_Scatter(l, self.x0, self.dt0, self.T, self.S, self.beta, self.sigma, self.phi)

        # 0. Geometric MLMC
        return MLMC_l_Geometric(l, 2, self.x0, self.dt0, self.T, self._F, self.sigma*xp.eye(2), self.phi)
