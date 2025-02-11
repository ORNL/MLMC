from typing import List, Dict, Any, Optional
from typing import TypeVar, Callable

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

def read_mcfr(name : Path) -> Dict[str,Any]:
    """ Read an MCFR file.
        Path -> {Str, Numeric}
    """
    with open(name, encoding="utf-8") as f:
        return json.load(f)

A = TypeVar("A")
B = TypeVar("B")
def zip_dict(fn : Callable[[List[A]], B], # [A] -> B
             ds : List[Dict[str,A]]) -> Dict[str,B]:
    """ Convert a list of dictionaries to
        a single dictionary.
    """
    if len(ds) == 0:
        return dict()
    return dict( (k, fn( [d[k] for d in ds] ))
                 for k in ds[0].keys()
               )

def plot_mcfrs(mcfrs : List[Dict[str,Any]],
               out : Optional[Path] = None) -> None:
    data = zip_dict(lambda x: x, mcfrs)

    dt_all = data['dt']
    N_all  = data['n_0']
    est_0_all = data['est_0']
    est_1_all = data['est_1']
    var_0_all = data['var_0']
    var_1_all = data['var_1']

    # Reorder the time steps
    ord = np.argsort(dt_all)
    # Careful, to use the array indexing (Matlab style...) you must get a numpy view of the list
    dt_all = np.array(dt_all)[ord]
    est_0_all = np.array(est_0_all)[ord]
    est_1_all = np.array(est_1_all)[ord]
    var_0_all = np.array(var_0_all)[ord]
    var_1_all = np.array(var_1_all)[ord]

    err_0 = np.sqrt(var_0_all)
    err_1 = np.sqrt(var_1_all)

    # Assume that the accuracy of MC:
    # error: the variance of MC(dt[0]]), this is at the most accurate level, where we assume that the bias is negligible w.r.t. the variance.
    # Print the variance
    print(dt_all)
    print(var_0_all)
    print(var_1_all)

    print(err_0)
    print(err_1)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
                                   sharex=True, sharey=True)
    ax0.set_xscale('log')
    ax1.set_xscale('log')

    # Note: if we are estimating the MGF,
    #    M(l) = < exp(l x) >
    # then the variance is related to the MGF
    #    < exp(lx)^2 > - <exp(lx)>^2 = M(2l) - M(l)^2

    # plt.semilogx(dt_all,est_0_all,linewidth=2)
    ax0.errorbar(dt_all,est_0_all,3*err_0, color='r', linewidth=2)
    ax0.errorbar(dt_all,est_0_all,err_0, color='b', linewidth=2)
    ax0.title.set_text(f'Time step vs Estimation 0 - MC {N_all[0]} particles')

    # plt.semilogx(dt_all,est_1_all,linewidth=2)
    ax1.errorbar(dt_all,est_1_all,3*err_1, color='r', linewidth=2)
    ax1.errorbar(dt_all,est_1_all,err_1, color='b', linewidth=2)
    ax1.title.set_text(f'Time step vs Estimation 1 - MC {N_all[0]} particles')
    if out is None:
        plt.show(block=False)
        input()
    else:
        plt.savefig(out)

def plot_mcfrs_n(mcfrs : List[Dict[str,Any]],
               out : Optional[Path] = None) -> None:
    data = zip_dict(lambda x: x, mcfrs)

    dt_all = data['dt']
    N_all  = data['n_0']
    est_0_all = data['est_0']
    est_1_all = data['est_1']
    var_0_all = data['var_0']
    var_1_all = data['var_1']

    # Reorder the time steps
    ord = np.argsort(N_all)
    # Careful, to use the array indexing (Matlab style...) you must get a numpy view of the list
    N_all = np.array(N_all)[ord]
    est_0_all = np.array(est_0_all)[ord]
    est_1_all = np.array(est_1_all)[ord]
    var_0_all = np.array(var_0_all)[ord]
    var_1_all = np.array(var_1_all)[ord]

    err_0 = np.sqrt(var_0_all)
    err_1 = np.sqrt(var_1_all)

    # Assume that the accuracy of MC:
    # error: the variance of MC(dt[0]]), this is at the most accurate level, where we assume that the bias is negligible w.r.t. the variance.
    # Print the variance
    print(dt_all)
    print(var_0_all)
    print(var_1_all)

    print(err_0)
    print(err_1)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
                                   sharex=True, sharey=True)
    ax0.set_xscale('log')
    ax1.set_xscale('log')

    # Note: if we are estimating the MGF,
    #    M(l) = < exp(l x) >
    # then the variance is related to the MGF
    #    < exp(lx)^2 > - <exp(lx)>^2 = M(2l) - M(l)^2

    # plt.semilogx(dt_all,est_0_all,linewidth=2)
    ax0.errorbar(N_all,est_0_all,3*err_0, color='r', linewidth=2)
    ax0.errorbar(N_all,est_0_all,err_0, color='b', linewidth=2)
    ax0.title.set_text(f'Number of particles vs Estimation 0 - MC dt={dt_all[0]}')

    # plt.semilogx(dt_all,est_1_all,linewidth=2)
    ax1.errorbar(N_all,est_1_all,3*err_1, color='r', linewidth=2)
    ax1.errorbar(N_all,est_1_all,err_1, color='b', linewidth=2)
    ax1.title.set_text(f'Number of particles vs Estimation 1 - MC dt={dt_all[0]}')
    if out is None:
        plt.show(block=False)
        input()
    else:
        plt.savefig(out)

def plot_distribution(r) -> None:
    # Plot the figures to make sure that the particles diffused
    fig = plt.figure()
    plt.plot(r[:,0,-1],r[:,1,-1], 'o',
        markeredgewidth = 0.5,
        markeredgecolor = 'black')
    plt.title('Distribution - x')
    plt.axis('equal')
    plt.show(block=False)