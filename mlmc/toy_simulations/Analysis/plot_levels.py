import numpy as np
import json
import matplotlib.pyplot as plt
import os
import write_ops

def load_json(file:str, tol:float):
    with open(file) as f:
        # More than one record per file...
        content = f.read()
        content = content.split("\n\n")

        data = []
        for c in content[0:-1]: # The last one is an empty string.
            dict = json.loads(c)

            err = dict["tol"]

            if err < tol:
                c = dict["c"]
                N = dict["n"]

                return (c, N, err)

def load_files(dir_MLMC:str, tol):
    labels_MLMC = []
    files_MLMC = []

    els = os.listdir(dir_MLMC)
    for el in els:
        # print(el)
        if el.endswith('.json'):
            files_MLMC.append( dir_MLMC + "/" + el )
            labels_MLMC.append( el.removesuffix(".json") )

    data = []
    for i in range(len(files_MLMC)):
        data.append( (labels_MLMC[i], load_json(files_MLMC[i], tol)) )
    return data

def crawl_dir(dir:str, tol):
    data = load_files(dir ,tol)
    # The search is a depth first search.
    for d in data:
        create_stem_plot_0(d[1], d[0], Block=False)
        input()

def create_stem_plot_0(data, title, Block=False):
    fig = plt.figure()
    N = data[1]
    
    L = len(N)
    xx = np.arange(L)

    cost=[0]*L
    if title=="Geom" or title=="Spring_F":
        cost[0] = N[0]
        for i in range(1,L):
            cost[i] = (2**i + 2**(i-1))*N[i]
    elif title=="Spring":
        cost[0] = N[0]
        for i in range(1,L):
            cost[i] = (2**(i+1))*N[i]

    plt.plot(xx, N, "-x", label=r'$N_l$', linewidth=2.4, markersize=16)
    plt.plot(xx, cost, "-o", label=r'$c_l/k_0$', linewidth=2.4, markersize=16)
    title = f"" + title + ", $\epsilon$={err:1.2e}"
    title = title.format(err=data[2])
    plt.title(title, fontsize=36)
    plt.legend(fontsize=36, loc=1)    

    ax = fig.get_axes()[0]
    # ax.set_xlabel('l', fontsize=48)
    ax.set_xticks(ticks=xx)
    ax.tick_params(axis='both', which='major', labelsize=48)
    ax.set_yticks([], minor=True)
    ax.set_yscale('log')
    plt.show(block=Block)
    

        
if(__name__=='__main__'):
    dir = 'results_toy_rich_bigstep/mu=1.0000/beta=2.50/MLMC/'
    crawl_dir(dir, tol=0.00005)