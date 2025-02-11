import numpy as np

from typing import Tuple

from mlmc.MLMC.MLMC_L import MLMC_l_AbstractFactory, MLMC_l

class MLMC_l_Simple(MLMC_l):

    def __init__(self, l:int):
        super().__init__()
        self.l = l

    def compute(self, N:int) -> Tuple[np.ndarray, float]:
        if self.l==0:
            # x0 = 0.2
            # dt = 1...
            # F = -x
            est = -0.2 + np.random.randn(1, N)
            cost = N
        else:
            # x0 = 0.2
            # dt = 1...
            dt1 = 1/(2**self.l)
            dt0 = 2*dt1
            s1 = np.sqrt(dt1)
            steps = 2**self.l
            dW = s1*np.random.randn( (steps,N) )
            
            est_1 = 0.2*np.ones(N)
            for i in np.arange(steps):
                est_1 += -est_1*dt1 + dW[i,:]
            
            est_0 = 0.2*np.ones(N)
            for i in np.arange( steps//2 ):
                est_0 += -est_0*dt0 + dW[2*i,:] + dW[2*i+1,:]
            est = est_1 - est_0

            cost = N* [2**(self.l-1)]*3
        return est, cost
    
class MLMC_l_SimpleFactory(MLMC_l_AbstractFactory):

    def __init__(self, beta:float, dt0:float, L_max:int, n:int, phi: any) -> any:
        super()
        self.beta = beta
        self.dt0 = dt0
        self.n = n
        self.phi = phi

    def create(self, l:int) -> MLMC_l_Simple:
        """ Create an instance of an object that computes the estimators for 
        level l. The concrete class MAY throw an error for l >= L_max, for some
        L_max
        """

        return MLMC_l_Simple(l)
