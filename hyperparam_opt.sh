#!/bin/bash

script_root="/data/scratch/scadavid/projects/code"
training_script="trainingLoop.py"

# charlie: gpu 2 and 8 are broken, 6 is not exactly broken

# make sure you're ssh'ed into a gpu and have activated the conda environment before running this script
CUDA_VISIBLE_DEVICES=0,1,2,3 python $training_script -lr 2e-4 -bs 8

wait
echo "All trainings complete!"