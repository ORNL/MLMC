from typing_extensions import Annotated
from typing import List, Optional, Tuple
from pathlib import Path
import json
import struct

import time
from math import pi, cos, sin

from mpi4py import MPI
import matplotlib.pyplot as plt

from mlmc.xp import xp
from mlmc.MLMC.MC import MC
from mlmc.MLMC.MLMC import MLMC

from mlmc.MD_simulations.MLMC_Scatter import MLMC_l_ScatterFactory, MLMC_Constraint
from mlmc.MD_simulations.ScatterFactory import ScatterFactory, Scatter
from mlmc.MD_simulations.EMFactory import EMFactory

from mlmc.MD_simulations.FilePlotter import read_mcfr, plot_mcfrs, plot_mcfrs_n

import typer
import numpy as np

app = typer.Typer()

# strategy = MLMC_Constraint.GEOM # MLMC_l_ScatterFactory.SPRING
# strategy = MLMC_Constraint.SPRING
# strategy = MLMC_Constraint.SPRING_CAP

class PhiProxy:
    def __init__(self,S:Scatter, const:float) -> None:
        self.const = const
        self.S = S

    def phi(self,x:xp.ndarray):
        E = self.S.E(x)
        emin = E.min()
        return xp.exp(self.const*emin) * xp.exp( self.const*(E-emin) )

    def exp_phi(self, E: xp.ndarray):
        emin = E.min()
        mean = xp.exp(self.const*emin) * xp.mean(xp.exp( self.const*(E-emin) ))
        var = xp.mean( (xp.exp(self.const*emin)*xp.exp( self.const*(E-emin) ) - mean)**2 )
        return mean, var

Rc = 3.3 # cutoff distance
theta = 60.0
def get_Scatter(D:float):
    # Set up the Scatter class

    return ScatterFactory(Rc,D,theta)

def load_D(file_D):
    # load D from the file_D
    with open(file_D, 'rb') as f:
        content = f.read()
        (D,) = struct.unpack('d',content)
    return D

@app.command()
def mlmc(
    dt : Annotated[float, typer.Argument(help="Time step of the coarsest simulation")],
    out : Annotated[Path, typer.Argument(help="Output file (json format)")],
    beta_1 : Annotated[float, typer.Option(help="beta (experiment)")] = 1.2,
    beta_2 : Annotated[float, typer.Option(help="beta (sanity check)")] = 1.1,
    T : Annotated[float, typer.Option(help="Time Horizon of the simulation [s]")] = 2.0,
    tol : Annotated[float, typer.Option(help="The desired tolerance eps")] = 1e-3,
    strategy : Annotated[MLMC_Constraint, typer.Option(help="MLMC Strategy")] = MLMC_Constraint.GEOM,
    D : Annotated[float, typer.Option(help="Distance between the scatters.")] = 4.0):
    """ Runs the tests with MLMC with beta_1 and beta_2, starting with dt and halving it as needed (L_max=4).
    """
    comm = MPI.COMM_WORLD

    dt0 = dt
    if comm.Get_rank() == 0:
        print('dt: ' + str(dt0) )
        print('D: ' + str(D) )

    # Compute the value of the Maximum Spring constant, if this is relevant.
    if strategy == MLMC_Constraint.SPRING_FIXED \
            or strategy == MLMC_Constraint.SPRING_AN_CAP:        
        K_max_D_min = 0.193336148747532077241118031452060677111148834228515625
        K_max_D_4 = 0.13172408938407909051448996251565404236316680908203125
        
        if D >= 4.0:
            K_max = K_max_D_4
        else:
            # Use a convex combination of the two spring constants.            
            D_min = Rc/sin(pi/3)
            t_big = (4.0 - D)
            t_small = (D - D_min)
            tt = t_big + t_small
            K_max = t_big*K_max_D_min + t_small*K_max_D_4
            K_max /= tt
    else:
        K_max = 0 # NOT RELEVANT

    comm.Barrier()

    S, x0 = get_Scatter(D)    

    #
    # RUN THE TEST WITH MLMC TO DISCOVER THE MEAN FIELD
    #
    pp = PhiProxy(S, beta_1-beta_2)
    mlmc_l_factory = MLMC_l_ScatterFactory(beta_1,x0,dt0,T,S, pp.phi, strategy=strategy,K_max=K_max)
    mlmc_run_0 = MLMC(mlmc_l_factory)

    tic = time.time()
    est_0, var_0 = mlmc_run_0.estimate(tol=tol,N0=1000, L_max=4)#, gamma=2.5)
    toc = time.time()
    time_0 = toc-tic
    if comm.Get_rank() == 0:
        print(f'Total time for the estimation: {time_0}')

    pp = PhiProxy(S, beta_2-beta_1)
    mlmc_l_factory = MLMC_l_ScatterFactory(beta_2,x0,dt0,T,S, pp.phi, strategy=strategy,K_max=K_max)
    mlmc_run_1 = MLMC(mlmc_l_factory)

    tic = time.time()
    est_1, var_1 = mlmc_run_1.estimate(tol=tol,N0=1000, L_max=4)#, gamma=2.5)
    toc = time.time()
    time_1 = toc-tic

    if comm.Get_rank() == 0:
        print(f'Total time for the estimation: {time_1}')

        print('est_0: ' + str(est_0) + ', err_est_0: ' + str(np.sqrt(var_0)))
        print('est_1: ' + str(est_1) + ', err_est_1: ' + str(np.sqrt(var_1)))
        print('sanity: ' + str(est_0*est_1))

        with open(out, "w", encoding="utf-8") as f:
            f.write( json.dumps({
                'dt': dt0,
                'n_0' : mlmc_run_0.Nl.tolist(),
                'est_0' : est_0,
                'var_0' : var_0,
                'time_0': time_0,
                'n_1' : mlmc_run_1.Nl.tolist(),
                'est_1' : est_1,
                'var_1' : var_1,
                'time_1': time_1,
                'check' : est_0*est_1}, indent=4))

    #fig = plt.figure()
    #plt.plot(dt_vec,est_0_v,linewidth=2)
    #plt.title('Time step vs Estimation - MLMC')
    #plt.show(block=False)

    #fig = plt.figure()
    #plt.plot(dt_vec,est_1_v,linewidth=2)
    #plt.title('Time step vs Estimation - MLMC')
    #plt.show(block=False)

    # fig = plt.figure()
    # plt.plot(dt_vec,sanity,linewidth=2)
    # plt.title('Time step vs Sanity check - MLMC')
    # plt.show(block=False)

@app.command()
def mc(
    dt : Annotated[float, typer.Argument(help="Time step for the simulation")],
    out : Annotated[str, typer.Argument(help="Output file (json format).")],
    beta_1 : Annotated[float, typer.Option(help="beta (experiment)")] = 1.2,
    beta_2 : Annotated[float, typer.Option(help="beta (sanity check)")] = 1.1,
    T : Annotated[float, typer.Option(help="Time Horizon of the simulation [s]")] = 2.0,
    n : Annotated[int, typer.Option(help="Number of walkers")] = 4096,
    D : Annotated[float, typer.Option(help="Distance between the scatters.")] = 4.0):
    """ Runs the tests on MC with beta_1 and beta_2 using n walkers.
    """
    comm = MPI.COMM_WORLD

    steps = int(xp.ceil(T/dt))
    if comm.Get_rank() == 0:
        print('Program is starting...')
        print('dt: ' + str(dt) )
        print('steps: ' + str(steps) )
        print('D: ' + str(D) )

    S, x0 = get_Scatter(D)

    # Initialize the integrators' factory
    sigma = 1
    i_f = EMFactory(beta_1,sigma,S)
    pp = PhiProxy(S, beta_1-beta_2)

    mc_run = MC(dt, n, x0, i_f, pp.phi)

    tic = time.time()
    est_0, var_0 = mc_run.estimate(steps)
    toc = time.time()
    time_0 = toc-tic

    if comm.Get_rank() == 0:
        print(f'Total time for the estimation: {time_0}')

        # r = mc_run.get_r()
        # print(r)

        # plt.figure()
        # plt.plot(r[:,0], r[:,1], '.', markersize=16)
        # plt.show(block=False)

        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(r[:,0])
        # plt.subplot(2,1,2)
        # plt.plot(r[:,1])
        # plt.show(block=True)

    i_f = EMFactory(beta_2,sigma,S)
    pp = PhiProxy(S, beta_2-beta_1)

    mc_run = MC(dt, n, x0, i_f, pp.phi)

    tic = time.time()
    est_1, var_1 = mc_run.estimate(steps)
    toc = time.time()
    time_1 = toc-tic

    if comm.Get_rank() == 0:
        print(f'Total time for the estimation: {time_1}')

        # r = mc_run.get_r()
        # print(r)

        # plt.figure()
        # plt.plot(r[:,0], r[:,1], '.', markersize=16)
        # plt.show(block=True)

        print('est_0: ' + str(est_0) + ', err_est_0: ' + str(np.sqrt(var_0/n)))
        print('est_1: ' + str(est_1) + ', err_est_1: ' + str(np.sqrt(var_1/n)))
        print('sanity: ' + str(est_0*est_1))

        with open(out, "w", encoding="utf-8") as f:
            f.write( json.dumps({
                'dt': dt,
                'n_0' : n,
                'est_0' : est_0,
                'var_0' : var_0/n,
                'time_0': time_0,
                'n_1' : n,
                'est_1' : est_1,
                'var_1' : var_1/n,
                'time_1': time_1,
                'check' : est_0*est_1}, indent=4))
