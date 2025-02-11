#!/bin/bash

# This script to collect runtime in the case of mu=1, beta=2.5
ranks=12

n_min=8192
n_max=536870912
n_mul=4

# Look at the tolerance obtained, use up to the same tolerance for MLMC.
declare -a type_vec=("0" "3" "5")

# Experiment 1
# declare -a tol_vec=("0.005" "0.001" "0.0005" "0.0001" "0.00005")
# declare -a mu_vec=("0.1250")
# declare -a beta_vec=("0.50")

# Experiment 2
# declare -a tol_vec=("0.005" "0.001" "0.0005" "0.0001" "0.00005")
# declare -a mu_vec=("1.0000")
# declare -a beta_vec=("2.50")

# Experiment 3
n_min=131072
declare -a tol_vec=("0.005" "0.001" "0.0005" "0.0002" "0.00008")
declare -a mu_vec=("2.0000")
declare -a beta_vec=("0.50")

# Experiment 4
# THE SPRING IS UNSTABLE, TRY to explain.
# n_min=131072
# n_max=134217728
# declare -a tol_vec=("0.006" "0.002" "0.0008" "0.0005" "0.00025")
# declare -a mu_vec=("2.0000")
# declare -a beta_vec=("5.00")

T=5.0

res=results_toy_rich_bigstep
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

         # 1. Fix n, run the MC as dt changes...
         # n=8388608

         # # Results to be put in this folder
         # folder=$folder_MC/n=$n
         # mkdir -p $folder

         # for dt in "${dt_vec[@]}"; do
         #    out=$folder/$dt.json
         #    echo $n, $dt
         #    mpiexec -np $ranks toy mc-test $dt $out --n $n --t $T --beta $beta --mu $mu
         # done

         # 2. Fix dt, run the MC as n changes...
         ## These are the maximum/minimum number of samples used
         declare -a dt_MC=("0.02" "0.005" "0.00125")

         for dt in "${dt_MC[@]}"; do
            ## Results to be put in this folder
            folder=$folder_MC/dt=$dt
            mkdir -p $folder

            for ((n=$n_min; n<=$n_max; n*=$n_mul)); do
               echo $n
               out=$folder/$n.json
               mpiexec -np $ranks toy mc-test $dt $out --n $n --t $T --beta $beta --mu $mu
            done
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
            dt=0.04
            for tol in "${tol_vec[@]}"; do
               echo $type_folder, $tol
               out=$folder/$tol.json
               mpiexec -np $ranks toy mlmc-test $dt $out --tol $tol --t $T --strategy $type_folder --mu $mu --beta $beta --lmax 5
               wait 
            done
         done
      fi
   done
done