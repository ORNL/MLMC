import numpy as np
import json
import os
import matplotlib.pyplot as plt
from cycler import cycler
import mlmc.MD_simulations.Analysis.write_ops as wo

def load_json(file:str):
    with open(file) as f:
        # More than one record per file...
        content = f.read()
        content = content.split("\n\n")

        data = []
        for c in content[0:-1]: # The last one is an empty string.
            dict = json.loads(c)

            c_0 = dict["c_0"]
            c_1 = dict["c_1"]
            tol_0 = dict["tol_0"]
            tol_1 = dict["tol_1"]
            time_0 = dict["time_0"]
            time_1 = dict["time_1"]

            data.append( (c_0, c_1, tol_0, tol_1, time_0, time_1) )

    return data

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
                create_figures_1(dir, block=False)
                input()
            else:
                crawl_dir(entry.path)

def create_figures_0(dir:str, block=True):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # fig = plt.figure

    ax0.set_yscale('log')
    ax1.set_yscale('log')
    ax0.set_xscale('log')
    ax1.set_xscale('log')

    data = get_data(dir)
    max_cost_0 = 0
    max_cost_1 = 0
    for d in data:
        cur = d[1]

        c_0_vec = []
        c_1_vec = []
        tol_0_vec = []
        tol_1_vec = []
        time_0_vec = [] 
        time_1_vec = []
        for tup in cur:
            c_0_vec.append( tup[0] )
            c_1_vec.append( tup[1] )
            tol_0_vec.append( tup[2] )
            tol_1_vec.append( tup[3] )
            time_0_vec.append( tup[4] )
            time_1_vec.append( tup[5] )

        indices = np.argsort(c_0_vec)
        c_0_vec = np.array(c_0_vec)[indices]
        c_1_vec = np.array(c_1_vec)[indices]
        tol_0_vec = np.array(tol_0_vec)[indices]
        tol_1_vec = np.array(tol_1_vec)[indices]
        time_0_vec = np.array(time_0_vec)[indices]
        time_1_vec = np.array(time_1_vec)[indices]

        ax0.plot(c_0_vec, tol_0_vec, '-s', label=d[0], linewidth=2.4, markersize=16)
        ax1.plot(c_1_vec, tol_1_vec, '-s', label=d[0], linewidth=2.4, markersize=16)

        max_cost_0 = max(max_cost_0, max(c_0_vec))
        max_cost_1 = max(max_cost_1, max(c_1_vec))

    ax0.plot( (0, max_cost_0), (5e-5, 5e-5), '--', label='$\epsilon=5e-5$')
    ax1.plot( (0, max_cost_1), (5e-5, 5e-5), '--', label='$\epsilon=5e-5$')

    ax0.set_ylabel('$\epsilon$', fontsize=48, rotation=0) 
    ax0.set_xlabel('estimated cost', fontsize=48)
    ax1.set_xlabel('estimated cost', fontsize=48)
    ax0.tick_params(axis='both', which='major', labelsize=48)
    ax1.tick_params(axis='both', which='major', labelsize=48)
    tx0 = ax0.xaxis.get_offset_text()
    tx1 = ax1.xaxis.get_offset_text()
    tx0.set_fontsize(48)
    tx1.set_fontsize(48)

    plt.legend(fontsize=32)

    index = dir.find("D=") # String of 2 chars.
    D_print = "$D=" + dir[index+2:index+8] + ", "
    index = dir.find("beta=") # String of 5 chars.
    beta_print = dir[index+5]
    beta_print = "\\beta_1=" + beta_print + ".2, \\beta_2=" + beta_print + ".1$" 

    title = D_print + beta_print
    plt.suptitle(title, fontsize=24)
    plt.show(block=block)
    return

def create_figures_1(dir:str, block=True):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # fig = plt.figure

    ax0.set_yscale('log')
    ax1.set_yscale('log')
    ax0.set_xscale('log')
    ax1.set_xscale('log')

    data = get_data(dir)
    max_time_0 = 0
    max_time_1 = 0
    for d in data:
        cur = d[1]

        c_0_vec = []
        c_1_vec = []
        tol_0_vec = []
        tol_1_vec = []
        time_0_vec = [] 
        time_1_vec = []
        for tup in cur:
            c_0_vec.append( tup[0] )
            c_1_vec.append( tup[1] )
            tol_0_vec.append( tup[2] )
            tol_1_vec.append( tup[3] )
            time_0_vec.append( tup[4] )
            time_1_vec.append( tup[5] )

        indices = np.argsort(c_0_vec)
        c_0_vec = np.array(c_0_vec)[indices]
        c_1_vec = np.array(c_1_vec)[indices]
        tol_0_vec = np.array(tol_0_vec)[indices]
        tol_1_vec = np.array(tol_1_vec)[indices]
        time_0_vec = np.array(time_0_vec)[indices]
        time_1_vec = np.array(time_1_vec)[indices]

        ax0.plot(time_0_vec, tol_0_vec, '-s', label=d[0], linewidth=2.4, markersize=16)
        ax1.plot(time_1_vec, tol_1_vec, '-s', label=d[0], linewidth=2.4, markersize=16)

        max_time_0 = max(max_time_0, max(time_0_vec))
        max_time_1 = max(max_time_1, max(time_1_vec))

    ax0.plot( (0, max_time_0), (5e-5, 5e-5), '--', label='$\epsilon=5e-5$')
    ax1.plot( (0, max_time_1), (5e-5, 5e-5), '--', label='$\epsilon=5e-5$')

    ax0.set_ylabel('$\epsilon$', fontsize=48, rotation=0)
    ax0.set_xlabel('run-time', fontsize=48)
    ax1.set_xlabel('run-time', fontsize=48)
    ax0.tick_params(axis='both', which='major', labelsize=48)
    ax1.tick_params(axis='both', which='major', labelsize=48)

    plt.legend(fontsize=32)

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
    dir_MLMC = dir + "/MLMC"
    labels_MLMC = []
    files_MLMC = []

    els = os.listdir(dir_MLMC)
    for el in els:
        # print(el)
        if el.endswith('.json'):
            files_MLMC.append( dir_MLMC + "/" + el )
            labels_MLMC.append( el.removesuffix(".json") )
    
    data = [ ]
    for i in range(len(files_MLMC)):
        data.append( (labels_MLMC[i], load_json(files_MLMC[i])) )
    
    return data

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
    col = [ col[5], col[1], col[2] , col[3], col[4], col[0] ] + col[6:-1]
    plt.rcParams['axes.prop_cycle'] = cycler(color=col)

    run(dir)