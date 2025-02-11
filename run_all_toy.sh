#!/bin/bash

# These are the MPI ranks that will run
ranks=12

# These are the MLMC strategies: 
#  0  Geometric MLMC
#  3  Adaptive Spring
#  5  Fixed Spring
declare -a type_vec=("0" "3" "5") 

# These is the range of MC particles. n_mul mens that the next experiment will have n*n_mul particles.
n_min=8192
n_max=4194304
n_mul=8

# Experiment 0
# INSTRUCTIONS: set mu, beta. RUN MC only with option 1.
# Look at the tolerance obtained, use up to the same tolerance for MLMC.
# declare -a mu_vec=("0.1250" "0.2500" "0.5000" "1.0000")
# declare -a beta_vec=("0.10" "0.25" "0.50" "1.00" "2.50" "5.00")

# The following are to tets for MLMC only
# 1
declare -a tol_vec=("0.005" "0.001" "0.0005")
declare -a mu_vec=("0.1250" "0.2500" "0.5000")
declare -a beta_vec=("0.10" "0.25" "0.50")

# 2
# declare -a tol_vec=("0.005" "0.001" "0.0005" "0.0002")
# declare -a mu_vec=("0.1250" "0.2500" "0.5000")
# declare -a beta_vec=("1.00" "2.50" "5.00")

# 3
# declare -a tol_vec=("0.005" "0.001" "0.0004")
# declare -a mu_vec=("1.0000")
# declare -a beta_vec=("0.10" "0.25" "0.50" "1.00" "2.50" "5.00")

# 4
# declare -a tol_vec=("0.005" "0.001" "0.0008")
# declare -a mu_vec=("2.0000")
# declare -a beta_vec=("0.10" "0.25" "0.50" "1.00" "2.50")

# 5 Very specific test, for 2, 5. This is to be run with MC, too
# n_min=32768
# n_max=16777216
# declare -a tol_vec=("0.005" "0.001" "0.0008")
# declare -a mu_vec=("2.0000")
# declare -a beta_vec=("5.00")

# 6 Another specific test, for 2, 0.5. This is the original test in the paper [Fang 2019].
# n_min=262144
# n_max=16777216
# declare -a tol_vec=("0.005" "0.001" "0.0005")
# declare -a mu_vec=("2.0000")
# declare -a beta_vec=("0.50")

# 7 A test with mu=1.5
# n_min=262144
# n_max=16777216
# declare -a tol_vec=("0.005" "0.001" "0.0008")
# declare -a mu_vec=("1.5000")
# declare -a beta_vec=("0.10" "0.25" "0.50" "1.00" "2.50" "5.00")

T=5.0

res=results_toy
for mu in "${mu_vec[@]}"; do
   res_mu=$res/mu=$mu
   echo $mu
   for beta in "${beta_vec[@]}"; do
      ## This is the result folder for the given beta.
      res_beta=$res_mu/beta=$beta

      if ! [ $1 -eq 0 ]; then
         echo \# computing MC
         ## This is the folder results for MC
         folder_MC=$res_beta/MC

         #  Fix dt, run the MC as n changes...         
         dt=0.00125

         ## Results to be put in this folder
         folder=$folder_MC/dt=$dt
         mkdir -p $folder

         for ((n=$n_min; n<=$n_max; n*=$n_mul)); do
            echo $n
            out=$folder/$n.json
            mpiexec -np $ranks toy mc-test $dt $out --n $n --t $T --beta $beta --mu $mu
         done
      fi

      if ! [ $1 -eq 1 ]; then
         echo \# computing MLMC
         # 3. As the tolerance varies, use MLMC. The variable type defines the kind of MLMC run, for now.

         ## This is the folder results for MC
         folder_MLMC=$res_beta/MLMC

         for type in "${type_vec[@]}"; do
            type_folder=Geom
            if [ $type -eq 3 ]; then
               type_folder=Spring
            elif [ $type -eq 5 ]; then
               type_folder=Spring_F
            fi

            folder=$folder_MLMC/$type_folder
            mkdir -p $folder
            dt=0.01
            for tol in "${tol_vec[@]}"; do
               echo $type_folder, $tol
               out=$folder/$tol.json
               mpiexec -np $ranks toy mlmc-test $dt $out --tol $tol --t $T --strategy $type_folder --mu $mu --beta $beta
               wait 
            done
         done
      fi
   done
done