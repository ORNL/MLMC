import numpy as np
import json
import os
import matplotlib.pyplot as plt
from cycler import cycler
import mlmc.MD_simulations.Analysis.write_ops as wo
import math 
import typer

def load_json(file:str):
    with open(file) as f:
        dict = json.loads(f.read())

        if( abs(dict["check"] - 1) > 0.01):
            if os.path.isfile(log_file):
                with open(log_file,'a') as f:
                    f.write("WARNING " + file + "sanity check failed.\n")
            else:
                print("WARNING " + file + "sanity check failed.")

        m_0 = dict["est_0"]
        m_1 = dict["est_1"]
        tol_0 = math.sqrt(dict["var_0"])
        tol_1 = math.sqrt(dict["var_1"])
    return m_0, m_1, tol_0, tol_1

def get_subdirs(dir: str):
    with os.scandir(dir) as it:
        # The search is in lexicographic order
        els = []
        for entry in it:
            if entry.is_dir():
                els.append(entry)
    els.sort(key=lambda dir: dir.name) # order everything in a lexicographic order.
    return els

def crawl_dir(dir:str):
    els = get_subdirs(dir)

    # The search is a depth first search.
    for entry in els:
        if entry.is_dir():
            if(entry.name == "MLMC"):
                create_figures_0(dir, block=False)
                input()
            else:
                crawl_dir(entry.path)

def create_figures_0(dir:str, block=True):
    fig = plt.figure()
    
    data = get_data(dir)

    max_tol = 0
    i = 0
    col = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for d in data:
        cur = d[1]

        m_0_vec = []
        m_1_vec = []
        tol_0_vec = []
        tol_1_vec = []
        for tup in cur:
            m_0_vec.append( tup[0] )
            m_1_vec.append( tup[1] )
            tol_0_vec.append( tup[2] )
            tol_1_vec.append( tup[3] )

        indices = np.argsort(tol_0_vec)
        tol_0_vec = np.array(tol_0_vec)[indices]
        tol_1_vec = np.array(tol_1_vec)[indices]
        m_0_vec = np.array(m_0_vec)[indices]
        m_2_vec = np.array(m_1_vec)[indices]

        # plt.plot(tol_0_vec, m_0_vec, '--*', color=col[i], label=d[0], linewidth=2.4, markersize=16)
        # plt.plot(tol_1_vec, m_1_vec, '--.', color=col[i], label=d[0], linewidth=2.4, markersize=16)
        plt.plot(tol_0_vec, m_1_vec*m_0_vec, '-s', color=col[i], label=d[0], linewidth=2.4, markersize=16)

        max_tol = max(max_tol, max(tol_0_vec))
        max_tol = max(max_tol, max(tol_1_vec))
        i+=1

    ax = fig.get_axes()[0]
    ax.set_xscale('log')

    ax.plot( (0, max_tol), (1, 1), '--')

    ax.set_xlabel('$\epsilon$', fontsize=48, rotation=0) 
    plt.ylim(0.99, 1.02)
    ax.tick_params(axis='both', which='major', labelsize=48)
    
    plt.legend(fontsize=48,loc=1)

    index = dir.find("D=") # String of 2 chars.
    D_print = "$D=" + dir[index+2:index+8] + ", "
    index = dir.find("beta=") # String of 5 chars.
    beta_print = dir[index+5]
    beta_print = "\\beta_1=" + beta_print + ".2, \\beta_2=" + beta_print + ".1$" 

    title = D_print + beta_print
    plt.suptitle(title, fontsize=24)
    plt.show(block=block)
    return

def get_data(dir:str):
    dir_MC = dir + "/MC"
    
    els = os.listdir(dir_MC)
    for el in els:
        # print(el)
        if (not el.endswith('.json')):            
            dir_MC = dir_MC + '/' + els[0]
            # consider only the first sub-directory, assume this is the relevant one.
            break

    dir_MLMC = dir + "/MLMC"
    labels_MLMC = []

    els = os.listdir(dir_MLMC)
    # The name of the directories should be the same as the labels...
    for el in els:
        # print(el)
        if el.endswith('.json'):            
            labels_MLMC.append( el.removesuffix(".json") )

    # Load MC data
    data_MC = []
    els = os.listdir(dir_MC)
    for el in els:
        if el.endswith('.json'):
            file = dir_MC + "/" + el
            data_MC.append(load_json(file))

    data = [ ('MC', data_MC) ]
    for l in labels_MLMC:
        dir = dir_MLMC + '/' + l
        data_MLMC = []
        els = os.listdir(dir)
        for el in els:
            if el.endswith('.json'):
                file = dir + "/" + el
                data_MLMC.append(load_json(file))
        data.append( (l, data_MLMC) )

    return data

app = typer.Typer()
@app.command()
def run(dir):
    """ It is necessary to have a directory with the results already in place before running this script.
    """
    crawl_dir(dir)

if(__name__=='__main__'):
    dir = 'results_MD'

    log_file = dir + '/log.txt'
    wo.crawl_dir(dir,log_file)

    # Make the colors like the ones in the double-well plot
    col = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # This sequence works, this script gets the folders in a different order
    col = [ col[0], col[5], col[1], col[2] , col[3], col[4] ] + col[6:-1]
    plt.rcParams['axes.prop_cycle'] = cycler(color=col)

    run(dir)