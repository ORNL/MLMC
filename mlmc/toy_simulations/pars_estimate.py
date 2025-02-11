from typing_extensions import Annotated
from pathlib import Path

import json

import mpi4py.MPI as MPI
import numpy as np

from mlmc.xp import xp
from mlmc.MLMC.MLMC import MLMC
from mlmc.ParallelExecution.ParallelExecutor import ParallelExecutor

from mlmc.toy_simulations.MLMC_Toy import MLMC_l_ToyFactory, MLMC_Constraint

import typer
import time

# Identity function.
def phi(x:xp.ndarray): 
    return x

app = typer.Typer()
@app.command()
def run(
    dt : Annotated[float, typer.Argument(help="Time step of the coarsest simulation")],
    out : Annotated[Path, typer.Argument(help="Output file (json format)")],
    beta : Annotated[float, typer.Option(help="beta (experiment)")] = 1,
    mu: Annotated[float, typer.Option(help="mu")] = 1,
    T : Annotated[float, typer.Option(help="Time Horizon of the simulation [s]")] = 5,
    tol : Annotated[float, typer.Option(help="The desired tolerance eps")] = 1e-3,
    strategy : Annotated[MLMC_Constraint, typer.Option(help="MLMC Strategy")] = MLMC_Constraint.GEOM):
    """ Runs the tests with MLMC on a double well with beta and mu.
    
    Tha maximum level is L=8, saves a file that contains the parameters of MLMC
        (alpha, beta, gamma) 
    as estimated with regression.
    
    It also saves the mean values and variance for each level, to estimate whether the exponential decay really happens.

    Adjust dt and the tolerance according to the requested (beta, mu), try to make it run for many levels...
    Depending on the parameters, the tolerance and the selected strategy, this test might run for a long time.
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
    est, var = mlmc_run.estimate(tol=tol,N0=2500, L_max=8)#, gamma=2.5)
    toc = time.time()
    runtime = toc-tic

    if comm.Get_rank() == 0:
        alpha, beta, gamma = mlmc_run.get_pars()

        mgb = 0.5*beta
        if(beta < gamma):
            print('beta is the smallest')
            print(f'half beta is {mgb}')
        else:
            print('gamma is the smallest')
            mgb = 0.5*gamma
            print(f'half gamma is {mgb}')
        
        cond = alpha < mgb
        if cond:
            print('The condition on alpha is not respected:')
            print(f'{alpha} < {mgb}')
        else:
            print('The condition on alpha is respected:')
            print(f'{alpha} >= {mgb}')

    
        print(f'Total time for the estimation: {runtime}')

        print('est_0: ' + str(est) + ', err_est: ' + str(np.sqrt(var)))

        with open(out, "w", encoding="utf-8") as f:
            f.write( json.dumps({
                'dt': dt0,
                'tol': np.sqrt(var),
                'N' : mlmc_run.Nl.tolist(),
                'm_l': mlmc_run.ml.tolist(),
                'V_l': mlmc_run.Vl.tolist(),
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'est' : est,
                'var' : var,
                'time': runtime }, indent=4))
            
if __name__ == '__main__':
    h=0.05
    
    # res='results_toy/est_pars/1_1_geom.out'
    # beta=1
    # mu=1
    
    res='results_toy/est_pars/1_2_geom.out'
    beta=1
    mu=2
    
    tol=10**(-3.5)
    # tol=1e-4
    # tol=10**(-4.2)
    # tol=10**(-4.5)
    
    run(h,res, tol=tol, mu=mu, beta=beta)