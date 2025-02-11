import numpy as np
import json
import os
import matplotlib.pyplot as plt
import re
import mlmc.toy_simulations.Analysis.write_ops as wo

from typing_extensions import Annotated
import typer
from cycler import cycler

# Now that the time step is variable, this is not enough. 
# Just select the timestep that you want to display in this particular plots.
MC_re = re.compile("dt=(\d|\.)*.json")
markers = ['s', '*', '^']
col = plt.rcParams['axes.prop_cycle'].by_key()['color']
col_MC = col[0]
col = [ col[1], col[2], col[3], col[4], col_MC ]
cycler( color=col )

plt.rcParams['axes.prop_cycle'] = cycler( color=col )

def load_json(file:str):
    with open(file) as f:
        # More than one record per file...
        content = f.read()
        content = content.split("\n\n")

        data = []
        for c in content[0:-1]: # The last one is an empty string.
            dict = json.loads(c)

            c = dict["c"]            
            tol = dict["tol"]            
            time = dict["time"]

            data.append( (c, tol, time) )

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
                input()
                break
            else:
                crawl_dir(entry.path)

def create_figures_0(dir:str, block=True):
    fig = plt.figure()
    print(dir)
    data = get_data(dir)
    max_cost = 0
    j=0
    for d in data:
        cur = d[1]

        c_vec = []        
        tol_vec = []        
        time_vec = [] 
        for tup in cur:
            if tup[0] > 0.9*10**7:
                c_vec.append( tup[0] )
                tol_vec.append( tup[1] )
                time_vec.append( tup[2] )

        indices = np.argsort(c_vec)
        c_vec = np.array(c_vec)[indices]
        tol_vec = np.array(tol_vec)[indices]
        time_vec = np.array(time_vec)[indices]

        if d[0].startswith('MC'):
            plt.plot(c_vec, tol_vec, '-' + markers[j], color=col_MC, label=d[0], linewidth=2.4, markersize=16)
            j+=1
        else:
            plt.plot(c_vec, tol_vec, '-^', label=d[0], linewidth=2.4, markersize=16)

        max_cost = max(max_cost, max(c_vec))

    ax = fig.get_axes()[0]
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # ax.plot( (0, max_cost), (1e-3, 1e-3), label='tol=1e-3')    
    ax.plot( (0, max_cost), (1e-4, 1e-4), '--', label='$\epsilon=1e-4$')
    
    ax.set_ylabel('$\epsilon$', fontsize=48, rotation=0)
    ax.set_xlabel('estimated cost', fontsize=48)
    ax.tick_params(axis='both', which='major', labelsize=48)
    ax.set_yticks([], minor=True)

    plt.legend(fontsize=36, loc=1)

    index_start = dir.find("mu=") # String of 2 chars.
    index_end = dir.find("beta=") # String of 5 chars.
    mu_print = dir[index_start+3:index_end-1]
    mu_print = "$\mu=" + str(float(mu_print)) # simple way to trim away the zeros.
    index_start = index_end+5
    beta_print = dir[index_start:]
    beta_print = str(float(beta_print)) # simple way to trim away the zeros.
    beta_print = ", \\beta=" + beta_print + "$"

    title = mu_print + beta_print
    plt.suptitle(title, fontsize=24)
    plt.show(block=block)
    return

def create_figures_1(dir:str, block=True):
    fig = plt.figure()

    data = get_data(dir)
    max_time = 0
    j=0
    for d in data:
        cur = d[1]

        c_vec = []        
        tol_vec = []    
        time_vec = [] 
        for tup in cur:
            if tup[0] > 0.9*10**7:
                c_vec.append( tup[0] )
                tol_vec.append( tup[1] )
                time_vec.append( tup[2] )

        indices = np.argsort(c_vec)
        c_vec = np.array(c_vec)[indices]        
        tol_vec = np.array(tol_vec)[indices]        
        time_vec = np.array(time_vec)[indices]

        if d[0].startswith('MC'):
            plt.plot(time_vec, tol_vec, '-' + markers[j], color=col_MC, label=d[0], linewidth=2.4, markersize=16)
            j+=1
        else:
            plt.plot(time_vec, tol_vec, '-^', label=d[0], linewidth=2.4, markersize=16)

        max_time = max(max_time, max(time_vec))

    ax = fig.get_axes()[0]
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.plot( (0, max_time), (1e-4, 1e-4), '--', label='$\epsilon=1e-4$')
    # ax.set_xticklabels([])

    ax.set_ylabel('$\epsilon$', fontsize=48, rotation=0) 
    ax.set_xlabel('run-time [s]', fontsize=48)
    ax.tick_params(axis='both', which='major', labelsize=48)
    
    plt.legend(fontsize=36)

    index_start = dir.find("mu=") # String of 2 chars.
    index_end = dir.find("beta=") # String of 5 chars.
    mu_print = dir[index_start+3:index_end-1]
    mu_print = "$\mu=" + str(float(mu_print)) # simple way to trim away the zeros.
    index_start = index_end+5
    beta_print = dir[index_start:]
    beta_print = str(float(beta_print)) # simple way to trim away the zeros.
    beta_print = ", \\beta=" + beta_print + "$"
    
    title = mu_print + beta_print

    plt.suptitle(title, fontsize=24)
    plt.show(block=block)
    return

def create_figures_2(dir:str, block=True):
    fig = plt.figure()

    data = get_data(dir)
    max_time = 0
    j=0
    for d in data:
        cur = d[1]

        c_vec = []        
        tol_vec = []    
        time_vec = [] 
        for tup in cur:
            if tup[0] > 0.9*10**7:
                c_vec.append( tup[0] )
                tol_vec.append( tup[1] )
                time_vec.append( tup[2] )

        indices = np.argsort(c_vec)
        c_vec = np.array(c_vec)[indices]        
        tol_vec = np.array(tol_vec)[indices]        
        time_vec = np.array(time_vec)[indices]

        if d[0].startswith('MC'):
            plt.plot(time_vec, tol_vec, '-' + markers[j], color=col_MC, label=d[0], linewidth=2.4, markersize=16)
            j+=1
        else:
            plt.plot(time_vec, tol_vec, '-^', label=d[0], linewidth=2.4, markersize=16)

        max_time = max(max_time, max(time_vec))

    ax = fig.get_axes()[0]
    ax.set_yscale('log')

    ax.plot( (0, max_time), (1e-4, 1e-4), '--', label='$\epsilon=1e-4$')
    # ax.set_xticklabels([])

    ax.set_ylabel('$\epsilon$', fontsize=36, rotation=0) 
    ax.set_xlabel('run-time [s]', fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=36)
    
    plt.legend(fontsize=36)

    index_start = dir.find("mu=") # String of 2 chars.
    index_end = dir.find("beta=") # String of 5 chars.
    mu_print = dir[index_start+3:index_end-1]
    mu_print = "$\mu=" + str(float(mu_print)) # simple way to trim away the zeros.
    index_start = index_end+5
    beta_print = dir[index_start:]
    beta_print = str(float(beta_print)) # simple way to trim away the zeros.
    beta_print = ", \\beta=" + beta_print + "$"
    
    title = mu_print + beta_print

    plt.suptitle(title, fontsize=12)
    plt.show(block=block)
    return

def get_data(dir:str):
    dir_MC = dir + "/MC"
    
    els = os.listdir(dir_MC)
    labels_MC = []
    files_MC = []
    h=[]
    for el in els:
        # print(el)
        if MC_re.match(el):
            files_MC.append(dir_MC + "/" + el)
            start = el.index('=')
            end = el.index('.j')
            hh = el[start:end]
            h.append(float(hh[1:]))
            labels_MC.append('MC, h' + hh)

    # Order the MC by the biggest h
    ord = np.flip(np.argsort(h))
    labels_MC = (np.array(labels_MC)[ord]).tolist()
    files_MC = (np.array(files_MC)[ord]).tolist()
    
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
    for i in range(len(files_MC)):
        data.append( (labels_MC[i], load_json(files_MC[i])) ) 
    
    for i in range(len(files_MLMC)):
        data.append( (labels_MLMC[i], load_json(files_MLMC[i])) )
    
    return data

app = typer.Typer()
@app.command()
def run(dir: Annotated[str, typer.Argument(help="The directory where the results are stored")]):
    """ It is necessary to have a directory with the results already in place before running this script.
    """
    log_file = dir + '/log.txt'
    wo.crawl_dir(dir,log_file)
    crawl_dir(dir)

if(__name__=='__main__'):
    dir = 'results_toy'
    run(dir)