from typing import Optional

import numpy as np
from mlmc.xp import xp

import struct

from mlmc.MLMC.MLMC_L import MLMC_l_AbstractFactory, MLMC_l

from mlmc.MLMC.MLMC_L_Geometric import MLMC_l_Geometric

from .F_double_well import F_double_well
from mlmc.Integration_SDE.EulerMaruyama import EulerMaruyama

from enum import Enum
class MLMC_Constraint(str, Enum):
    GEOM = "Geom"
    SPRING_FD = "Spring_FD"    
    SPRING_AD = "Spring"
    SPRING_FIXED = "Spring_F"

class MLMC_l_Toy(MLMC_l):
    _eps = 1e-6
    def __init__(self, l:int, x0:xp.ndarray, dt0:float, T:float, F:F_double_well, phi: any, K:float=-1, fixed:bool=False, analitical:bool=False):
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
        self.dt0 = dt0
        self.T = T
        self.phi = phi
        self.steps0 = int(np.floor(T/dt0))
        self._sigma = np.ones( (1,1) ) # Assume sigma is always 1
        self.gen = xp.random.default_rng(seed=None)
        self._F = F

        if fixed:
            self._K = self._K_fixed
            self._K_const = K
            self._cost_s = 0
        else:
            if analitical:
                self._K = self._K_analical
            else:
                self._K = self._K_unbound
            
            self._cost_s = 1

    def _K_analical(self,dY:xp.ndarray,Yc:xp.ndarray,Yf:xp.ndarray,FYc:xp.ndarray,FYf:xp.ndarray, N:int):
        # If the derivative is known, use it.
        dF = self._F.d2E(Yc)
        select = dF < 0
        dF[select] = 0
        return dF/2

    def _K_fixed(self,dY:xp.ndarray,Yc:xp.ndarray,Yf:xp.ndarray,FYc:xp.ndarray,FYf:xp.ndarray, N:int):
        return self._K_const*xp.ones((N,))

    # This function assigned to the relevant function at initialization....
    # def _K(self,dY:xp.ndarray,Yc:xp.ndarray,Yf:xp.ndarray,FYc:xp.ndarray,FYf:xp.ndarray, N:int):


    def _Force(self,x:xp.ndarray):        
        return self._F.dE(x)

    def _integrate(self, N):
        # Initialize the integrator
        em_in_0 = EulerMaruyama(self.dt0, self._Force, self._sigma )
        x = xp.zeros( (N,1) )
        x[:] = xp.copy(self.x0)

        for i in np.arange(self.steps0):
            x = em_in_0.step(x,self.gen)

        return self.phi(x), N*self.steps0

    def _spring(self, N):
        dt = self.dt0/(2**self.l) # Fine
        dt1 = dt/2

        k = int(np.floor(self.T/dt))
        k0 = int(np.floor(k/2))

        x0 = xp.zeros( N )
        x0[:] = self.x0

        ssqrt1 = self._sigma*np.sqrt(dt)

        # The Radon-Nikodym derivatives
        R_fine_exp = xp.zeros(N)
        R_coarse_exp = xp.zeros(N)

        # NOTE: the spring term is irrelevant at the first step, that's because Yf - Yc = x0 - x0 = 0
        Yf = x0
        Yc = x0

        k_cur = xp.zeros( N )
        Kf_hat = xp.zeros( N )

        # Every step of the coarse path is two steps of the fine one
        for j in range(k0):
            # Prepare the two random increments
            dW = ssqrt1*self.gen.standard_normal( (N,2) )
            dW0 = dW[:,0]
            dW1 = dW[:,1]

            # Update both the fine path and the coarse path
            Kc_hat = -Kf_hat # Used later in several instructions            

            FYc = self._Force(Yc)
            Yc_update = ( Kc_hat + FYc )*dt # Yf will be modified

            FYf = self._Force(Yf)            
            Yc = Yc + Yc_update + dW0            
            Yf = Yf + ( Kf_hat + FYf )*dt + dW0 # Yf modified

            # Update the Radon-Nikodym derivative (fine path only)
            R_fine_exp -= self._Radon_Nikodym_cur_exp(dW0, Kf_hat, dt1, N)

            # Update the spring constants
            dY = Yc - Yf
            # k_cur = 1.01*self._K(dY,Yc,Yf,FYc,FYf, N) # inflate the spring a little...

            # Number 2: update only the fine path and complete the update of the coarse path
            Kf_hat = k_cur*dY

            FYf = self._Force(Yf)
            Yc = Yc + Yc_update + dW1; # Yc modified.            
            Yf = Yf + ( Kf_hat + FYf )*dt + dW1
            # Update the Radon-Nikodym derivative (fine and coarse path)
            R_fine_exp -= self._Radon_Nikodym_cur_exp(dW1, Kf_hat, dt1, N)
            R_coarse_exp -= self._Radon_Nikodym_cur_exp(dW0 + dW1, Kc_hat, dt, N)

            # Update the spring
            dY = Yc - Yf

            k_cur = 1.01*self._K(dY,Yc,Yf,FYc,FYf,N)
            Kf_hat = k_cur*dY

        R_fine = xp.exp(R_fine_exp)
        R_coarse = xp.exp(R_coarse_exp)
        Pl = self.phi(Yf)*R_fine - self.phi(Yc)*R_coarse
        # j = xp.argmax(abs(Pl))
        # print('tsk_print_')
        # print( Pl[j] )
        # print( R_fine[j] )
        # print( R_coarse[j] )
        # print('_tsk_print')

        # Assume that the cost is 4 because there are 3 calls to _F and 1 to _K (_dF)
        # And we assue  that both F and dF have got the same wait.
        return Pl, N* (3+self._cost_s) *k0 

    def _Radon_Nikodym_cur_exp(self,dW:xp.ndarray, K_hat:xp.ndarray, dt_half:float, N:int):        
        K_hat = K_hat
        dKW = (dW + K_hat*dt_half)

        R_cur = K_hat*dKW
        return R_cur

    # This function assigned to the relevant function at initialization....
    # def compute(self, N:int) -> ( np.ndarray, int ):

class MLMC_l_ToyFactory(MLMC_l_AbstractFactory):

    def __init__(self, beta:float, mu:float, x0:xp.ndarray, dt0:float, T:float, phi: any, strategy:MLMC_Constraint=MLMC_Constraint.GEOM) -> any:
        super().__init__()
        self.mu = mu
        self.beta = beta
        self.dt0 = dt0
        self.x0 = x0
        self.T = T
        self.phi = phi        

        self.strategy = strategy
        
        self.K_fix = (mu*mu)*beta/2 # This is lambda/2

    def create(self, l:int) -> MLMC_l_Toy:
        if self.strategy == MLMC_Constraint.SPRING_FIXED:
            # 5. Version with fixed spring.
            # print(self.K_fix)
            return MLMC_l_Toy(l, self.x0, self.dt0, self.T, F_double_well(self.mu, self.beta), self.phi, K=self.K_fix, fixed=True)
        if self.strategy == MLMC_Constraint.SPRING_AD:
            # 3. Version with analitical spring.
            return MLMC_l_Toy(l, self.x0, self.dt0, self.T, F_double_well(self.mu, self.beta), self.phi, analitical=True)
        if self.strategy == MLMC_Constraint.SPRING_FD:
            # 1. Version with the spring
            return MLMC_l_Toy(l, self.x0, self.dt0, self.T, F_double_well(self.mu, self.beta), self.phi)

        # 0. Geometric MLMC
        sigma = np.ones( (1,1) )
        return MLMC_l_Geometric(l, 2, self.x0, self.dt0, self.T, F_double_well(self.mu, self.beta).dE, sigma, self.phi)
