import numpy as np
import json
import os
import matplotlib.pyplot as plt
import re
from cycler import cycler
import mlmc.MD_simulations.Analysis.write_ops as wo

MC_re = re.compile("dt=(\d|\.)*.json")

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
                create_figures_2(dir, block=False)
                create_figures_3(dir, block=False)
                input()
            else:
                crawl_dir(entry.path)

def create_figures_0(dir:str, block=True):
    fig = plt.figure()

    data = get_data(dir)
    max_cost_0 = 0
    for d in data:
        cur = d[1]

        c_0_vec = []
        tol_0_vec = []
        time_0_vec = [] 
        for tup in cur:
            c_0_vec.append( tup[0] )
            tol_0_vec.append( tup[2] )
            time_0_vec.append( tup[4] )

        indices = np.argsort(c_0_vec)
        c_0_vec = np.array(c_0_vec)[indices]
        tol_0_vec = np.array(tol_0_vec)[indices]
        time_0_vec = np.array(time_0_vec)[indices]

        plt.plot(c_0_vec, tol_0_vec, '-s', label=d[0], linewidth=3, markersize=16)

        max_cost_0 = max(max_cost_0, max(c_0_vec))

    ax = fig.get_axes()[0]
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot( (0, max_cost_0), (5e-5, 5e-5), '--', linewidth=2, label='$\epsilon=5e-5$')

    ax.set_ylabel('$\epsilon$', fontsize=48, rotation=0)
    ax.set_xlabel('estimated cost', fontsize=48)
    ax.tick_params(axis='both', which='major', labelsize=48)
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(48)
    plt.legend(fontsize=48)

    index = dir.find("D=") # String of 2 chars.
    D_print = "D=" + dir[index+2:index+8] + ", "
    index = dir.find("beta=") # String of 5 chars.
    beta_print = dir[index+5]
    beta_print = "\\beta=" + beta_print + ".2"

    title = "$" + D_print + beta_print + "$"
    plt.suptitle(title, fontsize=24)
    plt.show(block=block)

def create_figures_1(dir:str, block=True):
    fig = plt.figure()

    data = get_data(dir)
    max_time_0 = 0
    for d in data:
        cur = d[1]

        c_0_vec = []
        tol_0_vec = []
        time_0_vec = [] 
        for tup in cur:
            c_0_vec.append( tup[0] )
            tol_0_vec.append( tup[2] )
            time_0_vec.append( tup[4] )

        indices = np.argsort(c_0_vec)
        c_0_vec = np.array(c_0_vec)[indices]
        tol_0_vec = np.array(tol_0_vec)[indices]
        time_0_vec = np.array(time_0_vec)[indices]

        plt.plot(time_0_vec, tol_0_vec, '-s', label=d[0], linewidth=3, markersize=16)

        max_time_0 = max(max_time_0, max(time_0_vec))

    ax = fig.get_axes()[0]
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot( (0, max_time_0), (5e-5, 5e-5),  '--', linewidth=2, label='$\epsilon=5e-5$')
    
    ax.set_ylabel('$\epsilon$', fontsize=48, rotation=0)
    ax.set_xlabel('runtime', fontsize=48)
    ax.tick_params(axis='both', which='major', labelsize=48)
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(48)

    plt.legend(fontsize=48)

    index = dir.find("D=") # String of 2 chars.
    D_print = "D=" + dir[index+2:index+8] + ", "
    index = dir.find("beta=") # String of 5 chars.
    beta_print = dir[index+5]
    beta_print = "\\beta=" + beta_print + ".2"

    title = "$" + D_print + beta_print + "$" 
    plt.suptitle(title, fontsize=24)
    plt.show(block=block)
    return

def create_figures_2(dir:str, block=True):
    fig = plt.figure()

    data = get_data(dir)
    max_cost_0 = 0
    for d in data:
        cur = d[1]

        c_0_vec = []
        tol_0_vec = []
        time_0_vec = [] 
        for tup in cur:
            c_0_vec.append( tup[0] )
            tol_0_vec.append( tup[2] )
            time_0_vec.append( tup[4] )

        indices = np.argsort(c_0_vec)
        c_0_vec = np.array(c_0_vec)[indices]
        tol_0_vec = np.array(tol_0_vec)[indices]
        time_0_vec = np.array(time_0_vec)[indices]

        plt.plot(c_0_vec, tol_0_vec, '-s', label=d[0], linewidth=3, markersize=16)

        max_cost_0 = max(max_cost_0, max(c_0_vec))

    ax = fig.get_axes()[0]
    ax.set_yscale('log')
    ax.plot( (0, max_cost_0), (5e-5, 5e-5), '--', linewidth=2, label='$\epsilon=5e-5$')

    ax.set_ylabel('$\epsilon$', fontsize=48, rotation=0)
    ax.set_xlabel('estimated cost', fontsize=48)
    ax.tick_params(axis='both', which='major', labelsize=48)
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(48)

    plt.legend(fontsize=48)

    index = dir.find("D=") # String of 2 chars.
    D_print = "D=" + dir[index+2:index+8] + ", "
    index = dir.find("beta=") # String of 5 chars.
    beta_print = dir[index+5]
    beta_print = "\\beta=" + beta_print + ".2"

    title = "$" + D_print + beta_print + "$"
    plt.suptitle(title, fontsize=24)
    plt.show(block=block)

def create_figures_3(dir:str, block=True):
    fig = plt.figure()

    data = get_data(dir)
    max_time_0 = 0
    for d in data:
        cur = d[1]

        c_0_vec = []
        tol_0_vec = []
        time_0_vec = [] 
        for tup in cur:
            c_0_vec.append( tup[0] )
            tol_0_vec.append( tup[2] )
            time_0_vec.append( tup[4] )

        indices = np.argsort(c_0_vec)
        c_0_vec = np.array(c_0_vec)[indices]
        tol_0_vec = np.array(tol_0_vec)[indices]
        time_0_vec = np.array(time_0_vec)[indices]

        plt.plot(time_0_vec, tol_0_vec, '-s', label=d[0], linewidth=3, markersize=16)

        max_time_0 = max(max_time_0, max(time_0_vec))

    ax = fig.get_axes()[0]
    ax.set_yscale('log')
    ax.plot( (0, max_time_0), (5e-5, 5e-5),  '--', linewidth=2, label='$\epsilon=5e-5$')
    
    ax.set_ylabel('$\epsilon$', fontsize=48, rotation=0)
    ax.set_xlabel('run-time [s]', fontsize=48)
    ax.tick_params(axis='both', which='major', labelsize=48)
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(48)

    plt.legend(fontsize=48)

    index = dir.find("D=") # String of 2 chars.
    D_print = "D=" + dir[index+2:index+8] + ", "
    index = dir.find("beta=") # String of 5 chars.
    beta_print = dir[index+5]
    beta_print = "\\beta=" + beta_print + ".2"

    title = "$" + D_print + beta_print + "$" 
    plt.suptitle(title, fontsize=24)
    plt.show(block=block)

def get_data(dir:str):
    dir_MC = dir + "/MC"
    
    els = os.listdir(dir_MC)
    for el in els:
        # print(el)
        if MC_re.match(el):
            file_MC = dir_MC + "/" + el
            break
    
    dir_MLMC = dir + "/MLMC"
    labels_MLMC = []
    files_MLMC = []

    els = os.listdir(dir_MLMC)
    for el in els:
        # print(el)
        if el.endswith('.json'):
            files_MLMC.append( dir_MLMC + "/" + el )
            labels_MLMC.append( el.removesuffix(".json") )
    
    data = [ ('MC', load_json(file_MC)) ]
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
    col = [ col[0], col[5], col[1], col[2] , col[3], col[4] ] + col[6:-1]
    plt.rcParams['axes.prop_cycle'] = cycler(color=col)

    run(dir)