#!/bin/bash

training_script="trainingLoop.py"

# charlie: gpu 2,4 and 8 are broken, 6 is not exactly broken, 5 is weird ???
# S: gpu03 and 09 don't give me problems

# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-4 -bs 16 -w2 14.0 --dataset wsc --add_name 512-256-64-16 --layers 512,256,64,16 --add_name 512-256-64-16 &
CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-4 -bs 16 -w2 14.0 --dataset wsc --add_name 256-64-32 --layers 256,64,32
# CUDA_VISIBLE_DEVICES=2 python $training_script -lr 1e-4 -bs 16 -w2 14.0 --dataset wsc --add_name 256-64 --layers 256,64 &
# CUDA_VISIBLE_DEVICES=3 python $training_script -lr 1e-4 -bs 16 -w2 14.0 --dataset wsc --add_name 256-16 --layers 256,16 &
# should run 256,128,64 again

wait
echo "All trainings complete!"