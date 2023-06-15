#!/bin/bash

script_root="/data/scratch/scadavid/projects/code"
training_script="trainingLoop.py"

# charlie: gpu 2 and 8 are broken, 6 is not exactly broken
#gpu_list=("gpu03" "gpu03" "gpu03" "gpu03")
lr_list=("2e-4" "4e-4")
w2_list=("6.0" "9.0" "15.0" "30.0")

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

# make sure you're ssh'ed into a gpu and have activated the conda environment before running this script
CUDA_VISIBLE_DEVICES=0 python $training_script -lr 2e-4 -w1 1.0 -w2 6.0 &
CUDA_VISIBLE_DEVICES=1 python $training_script -lr 2e-4 -w1 1.0 -w2 9.0 &
CUDA_VISIBLE_DEVICES=2 python $training_script -lr 2e-4 -w1 1.0 -w2 15.0 &
CUDA_VISIBLE_DEVICES=3 python $training_script -lr 2e-4 -w1 1.0 -w2 30.0 &
CUDA_VISIBLE_DEVICES=4 python $training_script -lr 4e-4 -w1 1.0 -w2 6.0 &
CUDA_VISIBLE_DEVICES=5 python $training_script -lr 4e-4 -w1 1.0 -w2 9.0 &
CUDA_VISIBLE_DEVICES=6 python $training_script -lr 4e-4 -w1 1.0 -w2 15.0 &
CUDA_VISIBLE_DEVICES=7 python $training_script -lr 4e-4 -w1 1.0 -w2 30.0 &

wait
echo "All trainings complete!"

# then at end keep only file w highest f1? could actually have trainingLoop just return the max val f1 score and model as well and save the highest one that way