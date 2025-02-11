from typing_extensions import Annotated
from typing import List, Optional
from pathlib import Path
import json

import time

from mpi4py import MPI

from mlmc.xp import xp
from mlmc.MLMC.MC import MC
from mlmc.MLMC.MLMC import MLMC
from mlmc.MLMC.Factory import IntegratorFactory

import typer
import numpy as np

from .MLMC_Toy import MLMC_l_ToyFactory, MLMC_Constraint
from .F_double_well import F_double_well

from mlmc.Integration_SDE.EulerMaruyama import EulerMaruyama

app = typer.Typer()

# strategy = MLMC_Constraint.GEOM # MLMC_l_ScatterFactory.SPRING
# strategy = MLMC_Constraint.SPRING
# strategy = MLMC_Constraint.SPRING_CAP

# Identity function.
def phi(x:xp.ndarray): 
    return x
    
class IntFactory(IntegratorFactory):
    def __init__(self, F, sigma):
        self._F = F
        self._sigma = sigma

    def getIntegrator(self, dt:float):
        return EulerMaruyama(dt, self._F.dE, self._sigma)

@app.command()
def MLMC_test(
    dt : Annotated[float, typer.Argument(help="Time step of the coarsest simulation")],
    out : Annotated[Path, typer.Argument(help="Output file (json format)")],
    beta : Annotated[float, typer.Option(help="beta (experiment)")] = 1,
    mu: Annotated[float, typer.Option(help="mu (experiment)")] = 1,
    T : Annotated[float, typer.Option(help="Time Horizon of the simulation [s]")] = 5,
    tol : Annotated[float, typer.Option(help="The desired tolerance eps")] = 1e-3,
    Lmax : Annotated[float, typer.Option(help="The maximum admissible number of levels")] = 3,
    strategy : Annotated[MLMC_Constraint, typer.Option(help="MLMC Strategy")] = MLMC_Constraint.GEOM):
    """ Runs the tests with MLMC on a double well with beta and mu, starting with dt_max and halving it from one level tot he other, as needed.
        L_max=3
    """
    comm = MPI.COMM_WORLD

    dt0 = dt
    x0 = xp.array([0])

    #
    # RUN THE TEST WITH MLMC TO DISCOVER THE MEAN FIELD
    #
    mlmc_l_factory = MLMC_l_ToyFactory(beta, mu, x0, dt0, T, phi, strategy)
    mlmc_run = MLMC(mlmc_l_factory)

    tic = time.time()
    est, var = mlmc_run.estimate(tol=tol,N0=2500, L_max=Lmax)
    toc = time.time()
    runtime = toc-tic
    if comm.Get_rank() == 0:
        alpha, beta, gamma = mlmc_run.get_pars()

        print(f'Total time for the estimation: {runtime}')

        print('est_0: ' + str(est) + ', err_est: ' + str(np.sqrt(var)))

        with open(out, "w", encoding="utf-8") as f:
            f.write( json.dumps({
                'dt': dt0,
                'N' : mlmc_run.Nl.tolist(),
                'm_l': mlmc_run.ml.tolist(),
                'V_l': mlmc_run.Vl.tolist(),
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'est' : est,
                'var' : var,
                'time': runtime }, indent=4))
            
@app.command()
def MC_test(
    dt : Annotated[float, typer.Argument(help="Time step for the simulation")],
    out : Annotated[str, typer.Argument(help="Output file (json format).")],
    beta : Annotated[float, typer.Option(help="beta (experiment)")] = 1,
    mu: Annotated[float, typer.Option(help="mu (experiment)")] = 1,
    T : Annotated[float, typer.Option(help="Time Horizon of the simulation [s]")] = 5,
    n : Annotated[int, typer.Option(help="Number of walkers")] = 4096):
    """ Runs the tests on MC with beta and mu using n walkers.
    """
    comm = MPI.COMM_WORLD
    steps = int(xp.ceil(T/dt))
    x0 = xp.array([0])
    F = F_double_well(mu, beta)

    # Initialize the integrators' factory
    sigma = np.ones((1,1))
    i_f = IntFactory(F,sigma)

    mc_run = MC(dt, n, x0, i_f, phi)

    tic = time.time()
    est, var = mc_run.estimate(steps)
    toc = time.time()
    runtime = toc-tic

    if comm.Get_rank() == 0:
        print(f'Total time for the estimation: {runtime}')

        # r = mc_run.get_r()
        # print(r)

        # plt.figure()
        # plt.plot(r[:,0], r[:,1], '.', markersize=16)
        # plt.show(block=True)

        print('est: ' + str(est) + ', err_est: ' + str(np.sqrt(var/n)))

        with open(out, "w", encoding="utf-8") as f:
            f.write( json.dumps({
                'dt': dt,
                'N' : n,
                'est' : est,
                'var' : var/n,
                'time': runtime }, indent=4))