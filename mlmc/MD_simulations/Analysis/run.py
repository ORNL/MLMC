from typing_extensions import Annotated
import typer

import matplotlib.pyplot as plt
from cycler import cycler

import mlmc.MD_simulations.Analysis.write_ops as wo

import mlmc.MD_simulations.Analysis.check_times as ct
import mlmc.MD_simulations.Analysis.check_times_single as ct_single
import mlmc.MD_simulations.Analysis.check_times_noMC as ct_nMC
import mlmc.MD_simulations.Analysis.check_times_noMC_single as ct_nMC_single

app = typer.Typer()
@app.command()
def run(dir:Annotated[str, typer.Argument(help="The directory where the results are stored")], res:Annotated[int, typer.Argument(help="The kind of results to show")]=0):
    """ It is necessary to have a directory with the results already in place before running this script.

    Results type [res]:\n
        0   shows single cost and runtime (including MC)\n
        1   shows single cost and runtime (excluding MC)\n
        2   shows reciprocal runs cost and runtime (including MC)\n
        3   shows reciprocal runs and runtime (excluding MC)\n
    """
    log_file = dir + '/log.txt'
    wo.crawl_dir(dir,log_file)

    # Make the colors like the ones in the double-well plot
    col = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # This sequence works, this script gets the folders in a different order

    if res%2==0:
        col = [ col[0], col[5], col[1], col[2] , col[3], col[4] ] + col[6:-1]
    else:
        col = [ col[5], col[1], col[2] , col[3], col[4], col[0] ] + col[6:-1]
    plt.rcParams['axes.prop_cycle'] = cycler(color=col)

    if res == 0:
        ct_single.crawl_dir(dir)
    elif res == 1:
        ct_nMC_single.crawl_dir(dir)
    elif res == 2:
        ct.crawl_dir(dir)
    else:
        ct_nMC.crawl_dir(dir)
