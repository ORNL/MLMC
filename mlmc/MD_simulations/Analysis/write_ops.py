# This script is used to write a table for each technique
# The table contains the number of operations, who depend on the different technique,
# time step, number of samples (per level), etc...
# There are mainly 3 functions to this script:
# MC (simple), count how many steps to compute an MC estimation
# MLMC Geom (simple enough), count how many steps to compute a geometrical MLMC estimation
# MLMC Spring (unclear), count how many operations to compute an MLMC estimation with the spring.
# This last one is kinda euristic, since i never counted exactly how the operations compare, use the same
# Euristic used to return the cost in MLMC_l_Scatter functions.

import os
import os.path
import json
import math

T = 1.0 # Assumes that all the tests have been carried out for T=1.0
log_file = ''

def create_file(file:str):
    with open(file, 'w') as f:
        f.close()

def get_subdirs(dir: str):
    with os.scandir(dir) as it:
        # The search is in lexicographic order
        els = []
        for entry in it:
            if entry.is_dir():
                els.append(entry)
    els.sort(key=lambda dir: dir.name) # order everything in a lexicographic order.
    return els

def load_json(file:str):
    with open(file) as f:
        dict = json.loads(f.read())

        if( abs(dict["check"] - 1) > 0.01):
            if os.path.isfile(log_file):
                with open(log_file,'a') as f:
                    f.write("WARNING " + file + "sanity check failed.\n")
            else:
                print("WARNING " + file + "sanity check failed.")


        dt = dict["dt"]
        n_0 = dict["n_0"]
        n_1 = dict["n_1"]
        tol_0 = math.sqrt(dict["var_0"])
        tol_1 = math.sqrt(dict["var_1"])
        time_0 = dict["time_0"]
        time_1 = dict["time_1"]
    return dt, n_0, n_1, tol_0, tol_1, time_0, time_1

def crawl_dir(dir:str, log=None):
    # Create a log, should something go wrong.
    if(log is not None):
        create_file(log)

    els = get_subdirs(dir)

    # The search is a depth first search.
    for entry in els:
        if entry.is_dir():
            if(entry.name == "MC"):
                crawl_MC(entry.path)
            elif(entry.name == "MLMC"):
                crawl_MLMC(entry.path)
            else:
                crawl_dir(entry.path)

def crawl_MC(dir:str):
    els = get_subdirs(dir)

    for entry in els:
        file = entry.path + '.json'
        create_file(file)

        write_MC(entry.path, file)

def crawl_MLMC(dir:str):
    els = get_subdirs(dir)

    for entry in els:
        file = entry.path + '.json'
        create_file(file)
        if 'Spring' in entry.name:
            write_Spring(entry.path, file)
        else:
            write_Geom(entry.path, file)

def write_MC(dir:str, file):
    els = os.listdir(dir)
    els.sort() # order everything in a lexicographic order.

    for entry in els:
        if entry.endswith('.json'):
            ff = dir + '/' + entry
            dt, n_0, n_1, tol_0, tol_1, time_0, time_1 = load_json(ff)
            
            write_Cost(file, dt, n_0, n_1, MC_cost(dt,T,n_0), MC_cost(dt,T,n_1), tol_0, tol_1, time_0, time_1)

def write_Geom(dir:str, file):
    els = os.listdir(dir)
    els.sort() # order everything in a lexicographic order.

    for entry in els:
        if entry.endswith('.json'):
            ff = dir + '/' + entry
            dt, n_0, n_1, tol_0, tol_1, time_0, time_1 = load_json(ff)

            write_Cost(file, dt, n_0, n_1, Geom_cost(dt,T,n_0), Geom_cost(dt,T,n_1), tol_0, tol_1, time_0, time_1)

def write_Spring(dir:str, file):
    if dir == 'Spring_F':
        cost = Spring_F_cost
    else:
        cost = Spring_cost

    els = os.listdir(dir)
    els.sort() # order everything in a lexicographic order.

    for entry in els:
        if entry.endswith('.json'):
            ff = dir + '/' + entry
            dt, n_0, n_1, tol_0, tol_1, time_0, time_1 = load_json(ff)

            write_Cost(file, dt, n_0, n_1, cost(dt,T,n_0), cost(dt,T,n_1), tol_0, tol_1, time_0, time_1)

def write_Cost(file, dt, n_0, n_1, c_0, c_1, tol_0, tol_1, time_0, time_1):
    with open(file, 'a') as f:
        dumps = json.dumps({ "dt": dt, 
                            "n_0": n_0, 
                            "c_0": c_0, 
                            "tol_0": tol_0,
                            "time_0": time_0,
                            "n_1": n_1, 
                            "c_1": c_1,
                            "tol_1": tol_1,
                            "time_1": time_1, }, indent=0)
        f.write( dumps )
        f.write('\n\n')

def MC_cost(dt,T,n):
    return n*math.ceil(T/dt)

def Geom_cost(dt,T,n_vec):
    c = MC_cost(dt,T, n_vec[0])

    dt_0 = dt
    for n in n_vec[1:]:
        dt_1 = dt_0/2
        c += MC_cost(dt_0,T, n)
        c += MC_cost(dt_1,T, n)
        dt_0 = dt_1

    return c

def Spring_F_cost(dt,T,n_vec): # this is an euristic, confirm by looking at the times...
    c = MC_cost(dt,T, n_vec[0])

    dt_0 = dt
    for n in n_vec[1:]:
        dt_1 = dt_0/2
        c += 4*MC_cost(dt_0,T, n) # n*K_0 times the coarse R-N derivative
        dt_0 = dt_1

    return c

def Spring_cost(dt,T,n_vec): # this is an euristic, confirm by looking at the times...
    c = MC_cost(dt,T, n_vec[0])

    dt_0 = dt
    for n in n_vec[1:]:
        dt_1 = dt_0/2
        c += 5.5*MC_cost(dt_0,T, n) # n*K_0 times the coarse R-N derivative        
        dt_0 = dt_1

    return c

if(__name__=='__main__'):
    dir = 'results_MD'
    log_file = dir + '/log.txt'
    
    crawl_dir(dir, log=log_file)