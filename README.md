
# Multi-Level Monte Carlo Methods in Chemical Applications with Lennard-Jones Potentials and a Benchmark Problem

```#!
 ___    ___    _         ___    ___    ______
|   \  /   |  | |       |   \  /   |  / ___  \
| |\ \/ /| |  | |       | |\ \/ /| | | |   |_|
| | \__/ | |  | |       | | \__/ | | | |    _
| |      | |  | |_____  | |      | | | |___| |
|_|      |_|  |_______| |_|      |_|  \______/
```

Alberto Bocchinfuso, David M. Rogers

**LICENSE**
*This software is licensed under the BSD 3-Clause License. See the file* LICENSE

# 0. INSTALLATION

## 0.1. Venv

To create the virtual environment:

```#!/bin/bash
python -m venv v_env
```

Make sure to activate the environment with:

```#!/bin/bash
source v_env/bin/activate
```

__NOTE 0__ (python version): on certain systems

```#!/bin/bash
python3 -m venv v_env
```

might be required to use python3. The software is compatible with python 3.9 and above.

__NOTE 1__ (active environment): from this point on, the instructions assume that the virtual environment is active.
After creating the environment run the activation command above in this document. Make sure that the system acknowledges activation of the virtual environment;
"(v_env)" should appear somewhere in the command line, depending on the system. E.g.

```#!/bin/bash
(v_env) system:MLMC_repo user$ ...
```

## 0.2. Dependencies

To install the dependencies run:

```#!/bin/bash
poetry update
poetry install
```

__BONUS__: To run commands test-mlmc, test-scatter, ... without using poetry, install the current package with pip, using

```#!/bin/bash
pip install -e .
```

### 0.2.1. Extra dependencies

For use with GPUs

```#!/bin/bash
poetry install -E gpu-support
```

__NOTE__ (GPU faults): installation might fail because of several reasons, the most obvious because there is no Nvidia GPU currently available in the system.
Other reasons might include Nvidia drivers and CUDA softwares not correctly installed.

To be able to run analysis scripts (those who plot the results):

```#!/bin/bash
poetry install -E plots
```

# 1. RUN TESTS

To run examples

```#!/bin/bash
poetry run [test-mlmc, test-scatter, ...]
```

If commad the pip installation described above in this document has been executed without issues, write only

```#!/bin/bash
test-mlmc, test-scatter, ...
```

Section 2. gives instructions to run the code for the benchmark and the mean field examples.

# 2. RUN EXPERIMENTS

The ready made scripts run_all_toy.sh, run_toy_rich.sh and run_all_MD.sh work if the pip installation has been correctly executed.
The following assumes that it is possible to call toy (md-sim), without using poetry to call it.
If the pip installation was not executed, please, change the scripts to call poetry run toy (poetry run md-sim).

## 2.1. Toy (benchmark)

The files run_all_toy and file run_toy_rich present a set of different experiments that can be run.

### 2.1.1. run_all_toy

The following is the description of the file run_all_toy.sh
The tolerances for experiments 1-4 have been decided after running the MC simulations only and looking at the achieved accuracy.
Experiments 5-7 re-define the interval for MC, for performance reasons.

Comment all the experiments, but the one you want to run.
Experiment 0 runs MC tests in the range:

mu=0.125, ..., 1.0

For this experiment run

```#!/bin/bash
./run_all_toy 1
```

Experiment 1 can run both MC and MLMC. If MC has already been run using Experiment 0,

```#!/bin/bash
./run_all_toy 0
```

runs only the MLMC part. If the call is

```#!/bin/bash
./run_all_toy
```

then MC and MLMC will both run. The number of MC particles is decided as in Experiment 0.
Similar for the other experiments. If running experiments 5-7 with option 1 or 0, only MC or MLMC will be executed.

### 2.1.2. run_toy_rich

The file run_toy_rich.sh is similar to run_all_toy.sh, but it runs more tests on fewer (beta, mu) pairs.
There are 4 experiments in this case, with more tests for every (beta, mu) pair.

__NOTE__: The pictures in the paper have been obtained with this script.

## 2.2. Galton Board (Molecular Dynamics)

The file run_all_MD.sh runs the experiments for the Galton board case.
Called with 0, runs only MLMC, called with 1 runs only MC. Without any parameter, it runs both.

E.g.

```#!/bin/bash
./run_all_MD 1
```

runs only the MC part and doesn't save any MLMC results.

## 2.3. Analysis

Assuming that there are all or part of the results in a folder [DIR] it is possible to use the commands

```#!/bin/bash
(poetry run) toy-analysis [DIR]
(poetry run) MD-analysis [DIR] [0-3]
```

In the case of MD-analysis, it is possible to add 0-3 to print a subsets of results (e.g. no MC), please run

```#!/bin/bash
MD-analysis --help
```

Furthermore, the MD tests are run in pairs by the script run_all_MD.
For coupled tests as the one run by the provided script, it is known that

```#!
estimate_1*estimate_2 = 1
```

approximately. The command

```#!/bin/bash
(poetry run) MD-integrity [DIR]
```

prints pictures to show estimate_1*estimate_2 as a function of the tolerance. 
In the case of the provided results, the command is

```#!/bin/bash
(poetry run) MD-integrity results_MD
```

The ready made scripts save the results in the folders results_MD, results_toy_rich, results_toy.
Rady made results can be obtained by extracting the provided .tgz archives.

# 3. COMMAND LIST

```#!/bin/bash
test-mlmc       [TEST]
test-scatter    [TEST]

ex-toy-pars     [Example, see appendix A]

toy-analysis    [RESULTS - to print pictures, see 2.3]
md-analysis     [RESULTS - to print pictures, see 2.3]
md-integrity    [RESULTS - to print the integrity plots for MD, see 2.3]

toy             [RUN and SAVE the results using the specified parameters and technique.]
md-sim          [RUN and SAVE the results using the specified parameters and technique.]
```

## 3.1. Running experiments

The commands toy and md-sim are called from the scripts as in 2.1 and 2.2.
For further information on the commands themselves, run:

```#!/bin/bash
(poetry run) toy --help
(poetry run) toy mc --help
(poetry run) toy mlmc --help
(poetry run) md-sim --help
(poetry run) md-sim mc --help
(poetry run) md-sim mlmc --help
```

Alternatively, refer to the scripts to see how the commands are called there to obtain the results for relevant experiments.

# A. Example of MLMC use: estimate alpha, beta, gamma for Toy example

The file

mlmc/toy_simulations/pars_estimate.py

estiamates alpha, beta, gamma for the toy problem, given mu, beta etc.
The tighter the tolerance given, the more time needed, more samples used in the estimation.
The code also prints out whether the conditions in [Giles 2015], Theorem 2.1 about alpha, beta, gamma are respected.

Run 0. single core, slow - modify parameters in the main section of the script.

```#!/bin/bash
python mlmc/toy_simulations/pars_estimate.py
```

Run 1. multi core, fast - modify parameters in the main section of the script.

```#!/bin/bash
mpiexec -n [CORES] python mlmc/toy_simulations/pars_estimate.py
```

Run 2. single or multi core - submit parameters via command line.
Use the command ex-toy-pars with or without mpiexec.
E.g.

```#!/bin/bash
mpiexec -n 6 ex-toy-pars 0.01 results_toy/est_pars/025_2_spring_adp.out --beta 0.25 --mu 2.0 --tol 0.0001 --strategy Spring
```

# B. Matlab code

The folder Matlab contains a couple of scripts that are useful to understand how the tuning of the MD simulations was doing.
compute_D.m shows how the minimum and maximum D, given our assumptions are computed;
compute_K.m shows how the non-normalized maximum spring constant was computed.
The file MC_sim.conf is never used by the programs, it is used to collect the relevant values of D and K_max.
