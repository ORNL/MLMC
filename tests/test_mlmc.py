from tests.MLMC_Test import MLMC_l_Simple

import numpy as np
import mpi4py

from mlmc.MLMC.MC import MC
from mlmc.MLMC.Factory import IntegratorFactory

from mlmc.MLMC.MLMC import MLMC
from mlmc.MLMC.MLMC_L_Geometric import MLMC_l_Geometric, MLMC_l_AbstractFactory

from mlmc.Integration_SDE.EulerMaruyama import EulerMaruyama

import typer
app = typer.Typer()

class EM_Factory(IntegratorFactory):
    def __init__(self):
        self._sigma = np.ones((1,1))
    
    def getIntegrator(self, dt:float):
        return EulerMaruyama(dt, MLMC_l_GeomFactory.F, self._sigma)

class MLMC_l_TestFactory(MLMC_l_AbstractFactory):
    """ Everything is hard coded inside MLMC_Test.
    In this case, create just returns an object of the class computing the level l.
    """
    def create(self, l:int):
        return MLMC_l_Simple(l)
                                
class MLMC_l_GeomFactory(MLMC_l_AbstractFactory):
    """
    This class replicates MLMC_Test with the use of the ready made MLMC_l_Geometric.
    Just as in the case of MLMC_Test, here x0=0.2, dt0=T=1.0, sigma=1.0, phi=identity.
    In this class, this setup is in the __init__ method
    """
    def __init__(self, M:int=2):
        super().__init__()
        self._M = M
        self._x0 = np.array( [ 0.2 ] )
        self._dt0 = 1.0
        self._T = 1.0
        self._sigma = np.ones( (1,1) )

    # static
    def F(x: np.ndarray):
        return -x
    
    # static
    def phi(x: np.ndarray):
        return x

    def create(self, l:int):
        return MLMC_l_Geometric(l, self._M, self._x0, self._dt0, self._T, MLMC_l_GeomFactory.F, self._sigma, MLMC_l_GeomFactory.phi)



@app.command()
def main():
    """ This is a test of the code of MLMC with a very easy SDE. It uses both the ready made MLMC_l_Geometric class and a custom class in the file MLMC_Test,
        which is provided as an example code.
    """
    comm = mpi4py.MPI.COMM_WORLD
    r = comm.Get_rank()

    # 1. MC
    n = 2**20
    T = 1.0
    steps = 10000
    dt = T/steps # this is the same used in MLMC later
    x0 = np.array([0.2])
    i_f = EM_Factory()
    
    mc_run = MC(dt, n, x0, i_f, MLMC_l_GeomFactory.phi)
    est, var = mc_run.estimate(steps)

    if r == 0:
        print(str(est) + ', ' + str(var/n))
    
    # 2, 3, 4 - MLMC - with the same tolerance for all the variants used.
    # tol = 0.0001
    # tol = 0.0005
    tol = np.sqrt(var/n)

    # 2. Test with the Geometrical MLMC - built in class MLMC_l_Geometric.
    # Just write the Factory class MLMC_l_GeomFactory.
    # In this case, all the necessary information (x0, dt0, etc...) is hard coded
    # in the class init method.
    mlmc_geom_factory = MLMC_l_GeomFactory()
    mlmc_geom_run = MLMC(mlmc_geom_factory)

    est, var = mlmc_geom_run.estimate(tol)
    
    if r == 0:
        print('Geometrical MLMC estimate (example implementation - tests.MLMC_l_Simple): ' + str(est) + ', ' + str(var))

    # 3. Test with the MLMC_l_Simple in MLMC_Test.
    # Just write the Factory class MLMC_l_TestFactory.
    # In this case, all the necessary information (x0, dt0, etc...) is hard coded
    # in the code of MLMC_l_Simple.
    mlmc_test_factory = MLMC_l_GeomFactory()
    mlmc_test_run = MLMC(mlmc_test_factory)

    est, var = mlmc_test_run.estimate(tol)
    
    if r == 0:
        print('Geometrical MLMC estimate (using mlmc.MLMC_l_Geometric): ' + str(est) + ', ' + str(var))

    # 4. Like point 2, but run it with M=4 just to try.
    M=4
    mlmc_geom_4_factory = MLMC_l_GeomFactory(M)
    mlmc_geom_4_run = MLMC(mlmc_geom_4_factory)

    est, var = mlmc_geom_4_run.estimate(tol)
    
    if r == 0:
        print('Geometrical MLMC estimate using M=4: ' + str(est) + ', ' + str(var))    

if __name__ == '__main__':
    main()