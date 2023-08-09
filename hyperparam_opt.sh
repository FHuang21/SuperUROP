#!/bin/bash

training_script="trainingLoop.py"

# charlie: gpu 2,4 and 8 are broken, 6 is not exactly broken, 5 is weird ???
# S: gpu03 and 09 don't give me problems
# note: when running multiple instances, add '&' after each line

## SDS Regression
# CUDA_VISIBLE_DEVICES=0 python3 $training_script --task regression -lr 1e-4 -bs 4 --num_epochs 50 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=1 python3 $training_script --task regression -lr 1e-4 -bs 8 --num_epochs 50 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=2 python3 $training_script --task regression -lr 1e-4 -bs 16 --num_epochs 50 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=3 python3 $training_script --task regression -lr 1e-4 -bs 32 --num_epochs 50 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &

# wait
# CUDA_VISIBLE_DEVICES=0 python3 $training_script --task regression -lr 1e-4 -bs 4 --num_epochs 100 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=1 python3 $training_script --task regression -lr 1e-4 -bs 8 --num_epochs 100 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=2 python3 $training_script --task regression -lr 1e-4 -bs 16 --num_epochs 100 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=3 python3 $training_script --task regression -lr 1e-4 -bs 32 --num_epochs 100 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &

# wait
# CUDA_VISIBLE_DEVICES=0 python3 $training_script --task regression -lr 1e-4 -bs 4 --num_epochs 30 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=1 python3 $training_script --task regression -lr 1e-4 -bs 8 --num_epochs 30 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=2 python3 $training_script --task regression -lr 1e-4 -bs 16 --num_epochs 30 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=3 python3 $training_script --task regression -lr 1e-4 -bs 32 --num_epochs 30 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &

# wait
# CUDA_VISIBLE_DEVICES=0 python3 $training_script --task regression -lr 1e-4 -bs 4 --num_epochs 70 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=1 python3 $training_script --task regression -lr 1e-4 -bs 8 --num_epochs 70 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=2 python3 $training_script --task regression -lr 1e-4 -bs 16 --num_epochs 70 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &
# CUDA_VISIBLE_DEVICES=3 python3 $training_script --task regression -lr 1e-4 -bs 32 --num_epochs 70 --num_classes 1 --num_heads 4 --dataset shhs2 --label dep --simon_model --add_name _tune --control &

## other drug classification

## best benzo params:
#CUDA_VISIBLE_DEVICES=0 python3 $training_script --task multiclass -lr 2e-4 -w 1.0,8.0 -bs 16 --num_epochs 20 --num_classes 2 --num_heads 4 --dropout 0.4 --dataset shhs2 --label benzo --simon_model --full_set --add_name _fullset

# thyroid
CUDA_VISIBLE_DEVICES=0 python3 $training_script --task multiclass -lr 1e-4 -w 1.0,10.0 -bs 16 --num_epochs 100 --num_classes 2 --num_heads 4 --dropout 0.4 --dataset shhs2 --label thyroid --simon_model --add_name _tune3 &
CUDA_VISIBLE_DEVICES=1 python3 $training_script --task multiclass -lr 2e-4 -w 1.0,10.0 -bs 16 --num_epochs 100 --num_classes 2 --num_heads 4 --dropout 0.4 --dataset shhs2 --label thyroid --simon_model --add_name _tune3 &
CUDA_VISIBLE_DEVICES=2 python3 $training_script --task multiclass -lr 8e-5 -w 1.0,10.0 -bs 16 --num_epochs 100 --num_classes 2 --num_heads 4 --dropout 0.4 --dataset shhs2 --label thyroid --simon_model --add_name _tune3 &
CUDA_VISIBLE_DEVICES=3 python3 $training_script --task multiclass -lr 6e-5 -w 1.0,10.0 -bs 16 --num_epochs 100 --num_classes 2 --num_heads 4 --dropout 0.4 --dataset shhs2 --label thyroid --simon_model --add_name _tune3 &
# epochs / learning rate

wait
echo "All trainings complete!"