#!/bin/bash

script_root="/data/scratch/scadavid/projects/code"
training_script="trainingLoop.py"

# charlie: gpu 2 and 8 are broken, 6 is not exactly broken, 5 is weird ???
#gpu_list=("gpu03" "gpu03" "gpu03" "gpu03")
# lr_list=("2e-4" "4e-4")
# w2_list=("6.0" "9.0" "15.0" "30.0")

# make sure you're ssh'ed into a gpu and have activated the conda environment before running this script
#CUDA_VISIBLE_DEVICES=0,1,2,3 python $training_script -lr 1e-4 &
#CUDA_VISIBLE_DEVICES=4,5,6,7 python $training_script -lr 1e-4 --dataset shhs1 &
#CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-4 &

# # EEG TS
# CUDA_VISIBLE_DEVICES=0,1 python $training_script -lr 1e-4 -bs 2 &
# CUDA_VISIBLE_DEVICES=2,3 python $training_script -lr 1e-4 -bs 2 --dataset shhs2 &
#thing:
CUDA_VISIBLE_DEVICES=0,1,2,3 python $training_script -lr 1e-4 -bs 4 #&
#CUDA_VISIBLE_DEVICES=4,5,6,7 python $training_script -lr 1e-4 -bs 4 --datatype spec --input eeg #&

# # BB TS
# CUDA_VISIBLE_DEVICES=4,5 python $training_script -lr 1e-4 -bs 2 --data_source bb --num_channel 2016 &
# CUDA_VISIBLE_DEVICES=6,7 python $training_script -lr 1e-4 -bs 2 --dataset shhs2 --data_source bb --num_channel 2016 &

# EEG SPEC
# CUDA_VISIBLE_DEVICES=0 python $training_script -lr 1e-4 -bs 2 --datatype spec &
# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 2e-4 -bs 2 --datatype spec &
# CUDA_VISIBLE_DEVICES=2 python $training_script -lr 1e-4 -bs 2 --dataset shhs2 --datatype spec &
# CUDA_VISIBLE_DEVICES=3 python $training_script -lr 2e-4 -bs 2 --dataset shhs2 --datatype spec &

# BB SPEC
# CUDA_VISIBLE_DEVICES=4 python $training_script -lr 1e-4 -bs 2 --data_source bb --datatype spec &
# CUDA_VISIBLE_DEVICES=5 python $training_script -lr 2e-4 -bs 2 --data_source bb --datatype spec &
# CUDA_VISIBLE_DEVICES=6 python $training_script -lr 1e-4 -bs 2 --dataset shhs2 --data_source bb --datatype spec &
# CUDA_VISIBLE_DEVICES=7 python $training_script -lr 2e-4 -bs 2 --dataset shhs2 --data_source bb --datatype spec &


wait
echo "All trainings complete!"

# then at end keep only file w highest f1? could actually have trainingLoop just return the max val f1 score and model as well and save the highest one that way


# for i in {0..3}
# do 
#     gpu=${gpu_list[i]}
#     lr=${lr_list[i]}
#     w1=1.0
#     w2=${w2_list[i]}
#     echo "Connecting to $gpu"
#     ssh scadavid@netmit$gpu "source /data/scratch/scadavid/miniconda3/bin/activate; conda activate research && \
#         cd $script_root && CUDA_VISIBLE_DEVICES=$i python $training_script -lr $lr -w1 $w1 -w2 $w2" &
#     echo "Running on $gpu"
# done

# the following code doesn't work and is garbage:
# gpu="gpu05"
# w1=1.0
# remote_ssh_command="ssh scadavid@netmit$gpu"  # SSH command
# ssh_session=$($remote_ssh_command)
# #$ssh_session "source /data/scratch/scadavid/miniconda3/bin/activate; conda activate research && cd $script_root"

# i=0
# for lr in "${lr_list[@]}"
# do
#     for w2 in "${w2_list[@]}"
#     do
#         $ssh_session "source /data/scratch/scadavid/miniconda3/bin/activate; conda activate research && cd $script_root && CUDA_VISIBLE_DEVICES=$i python $training_script -lr $lr -w1 $w1 -w2 $w2" &
#         echo "Running on core #$i"
#         ((i++))
#     done
# done