#!/bin/bash

## This parameters to define the MPI run...
ranks=12

## Declare the array variables you need.
declare -a tol_vec=("0.005" "0.001" "0.0005" "0.0001" "0.00005")
declare -a dt_vec=("0.001" "0.0005" "0.0001")
declare -a type_vec=("0" "3" "4" "5")
declare -a beta_vec=("1" "2" "4")
declare -a D_vec=("3.810511776651530180259896951611153781414031982421875" "4.00000" "5.715767664977295048345240502385422587394714355468750")

## Declare the time interval
T=1.0

# Define maximum/minimum number of particles for MonteCarlo.
n_min=8192
n_max=1048576

## This is the result folder
res=results_MD
for D in "${D_vec[@]}"; do
   echo D is ${D:0:6}

   res_D=$res/D=${D:0:6}

   for beta in "${beta_vec[@]}"; do
      ## This is the beta for the test.
      beta_1=$beta.2
      beta_2=$beta.1

      echo beta is $beta: beta_1=$beta_1, beta_2=$beta_2

      ## This is the result folder for the given beta.
      res_beta=$res_D/beta=$beta

      if ! [ $1 -eq 0 ]; then
         echo computing MC before MLMC
         ## This is the folder results for MC
         folder_MC=$res_beta/MC
         
         mkdir -p $folder_MC

         # 1. Fix n, run the MC as dt changes...
         # n=16384 # 65536 # 4096 # 65536 # 1048576 # 262144

         # Results to be put in this folder
         # folder=$folder_MC/n=$n
         # mkdir -p $folder

         # for dt in "${dt_vec[@]}"; do
         #    echo MC with n=$n, dt=$dt
         #    mpiexec -np $ranks sim mc $dt $folder/$dt.json --n $n --t $T --beta-1 $beta_1 --beta-2 $beta_2 --d $D &
         #    wait
         # done

         # mlmc plot-mcfr -o $folder/out.pdf $folder/*.json

         # 2. Fix dt, run the MC as n changes...
         dt=0.0000625

         ## Results to be put in this folder
         folder=$folder_MC/dt=$dt
         mkdir -p $folder         

         for ((n=$n_min; n<=$n_max; n*=2)); do
            echo MC with n=$n, dt=$dt
            mpiexec -np $ranks md-sim mc $dt $folder/$n.json --n $n --t $T --beta-1 $beta_1 --beta-2 $beta_2 --d $D &
            wait
         done

         md-sim plot-mcfr-n -o $folder/out.pdf $folder/*.json
      fi

      if ! [ $1 -eq 1 ]; then
         # 3. As the tolerance varies, use MLMC. The variable type defines the kind of MLMC run, for now.

         ## This is the folder results for MC
         folder_MLMC=$res_beta/MLMC
         
         mkdir -p $folder_MLMC

         for type in "${type_vec[@]}"; do
            type_folder=Geom
            if [ $type -eq 1 ]; then
               type_folder=Spring_FD
               # echo Spring with finite differences
            elif [ $type -eq 3 ]; then
               type_folder=Spring
               # echo Spring
            elif [ $type -eq 4 ]; then
               type_folder=Spring_Cap
               # echo Spring Cap
            elif [ $type -eq 5 ]; then
               type_folder=Spring_F
               # echo Spring with a fixed value
            # else
               # echo Geometric MLMC
            fi

            folder=$folder_MLMC/$type_folder
            mkdir -p $folder

            dt=0.001
            for tol in "${tol_vec[@]}"; do
               echo MLMC "("$type_folder")", dt=$dt, tol=$tol
               mpiexec -np $ranks  md-sim mlmc $dt $folder/$tol.json --tol $tol --t $T --strategy $type_folder --beta-1 $beta_1 --beta-2 $beta_2 --d $D &
               wait
            done
         done
      fi
   done
done