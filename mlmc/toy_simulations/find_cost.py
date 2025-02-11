import time

from mlmc.xp import xp
from mlmc.MLMC.MC import MC
from mlmc.MLMC.MLMC import MLMC
from mlmc.MLMC.Factory import IntegratorFactory

import typer
import numpy as np

from mlmc.toy_simulations.MLMC_Toy import MLMC_l_ToyFactory, MLMC_Constraint
from mlmc.toy_simulations.F_double_well import F_double_well

from mlmc.Integration_SDE.EulerMaruyama import EulerMaruyama

# Identity function.
def phi(x:xp.ndarray): 
    return x
    
class IntFactory(IntegratorFactory):
    def __init__(self, F, sigma):
        self._F = F
        self._sigma = sigma

    def getIntegrator(self, dt:float):
        return EulerMaruyama(dt, self._F.dE, self._sigma)

beta = 1
mu = 1
T = 5.0
dt0 = 0.1

x0 = xp.array([0])


strategy = [ MLMC_Constraint.GEOM, MLMC_Constraint.SPRING_AD, MLMC_Constraint.SPRING_FIXED ]

for s in strategy:
    print(s)
    mlmc_l_factory = MLMC_l_ToyFactory(beta, mu, x0, dt0, T, phi, s)
    N = 1000000
    for l in range(6):
        mlmc_l = mlmc_l_factory.create(l)
        tic = time.time()
        a, c = mlmc_l.compute(N)
        toc = time.time()
        runtime = toc-tic
        print(str(l) + ', ' + str(c) + ', ' + str(runtime))

    print()
    print()
