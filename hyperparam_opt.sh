#!/bin/bash

training_script="trainingLoop.py"

# charlie: gpu 2,4 and 8 are broken, 6 is not exactly broken, 5 is weird ???
# S: gpu03 and 09 don't give me problems
# note: when running multiple instances, add '&' after each line

# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-4 -bs 16 -w2 14.0 --dataset wsc --add_name 512-256-64-16 --layers 512,256,64,16 --add_name 512-256-64-16 &
# CUDA_VISIBLE_DEVICES=4 python $training_script -lr 1e-5 -w 1.0,45.0,5.0,20.0 --num_classes 4 -bs 16 --dataset wsc &
# CUDA_VISIBLE_DEVICES=5 python $training_script -lr 1e-4 -w 1.0,45.0,5.0,20.0 --num_classes 4 -bs 16 --dataset wsc &
# CUDA_VISIBLE_DEVICES=6 python $training_script -lr 1e-3 -w 1.0,45.0,5.0,20.0 --num_classes 4 -bs 16 --dataset wsc &
# CUDA_VISIBLE_DEVICES=7 python $training_script -lr 1e-2 -w 1.0,45.0,5.0,20.0 --num_classes 4 -bs 16 --dataset wsc &
# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-3 -w 1.0,45.0,5.0,20.0 --num_classes 4 --dataset wsc --pretrained --model_path /data/scratch/scadavid/projects/data/models/encoding/wsc/class_2/eeg/lr_0.0001_w1_1.0_w2_14.0_posf1_0.7_antidep_256,64,16.pt

##combined binary antidepressant classification HP tuning
# CUDA_VISIBLE_DEVICES=0 python $training_script -lr 2e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary &
# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 4e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary &
# CUDA_VISIBLE_DEVICES=2 python $training_script -lr 8e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary &
# CUDA_VISIBLE_DEVICES=3 python $training_script -lr 8e-5 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary &
# CUDA_VISIBLE_DEVICES=4 python $training_script -lr 6e-5 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary &

## wsc depression regression
# CUDA_VISIBLE_DEVICES=0 python $training_script -lr 5e-5 -w 1.0,14.0 --num_classes 1 --dataset wsc --label dep --task regression --num_epochs 500 --add_name _thr_44 & #&
# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-4 -w 1.0,14.0 --num_classes 1 --dataset wsc --label dep --task regression --num_epochs 500 --add_name _thr_44 &
#CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-3 -w 1.0,14.0 --num_classes 1 --dataset wsc --label dep --task regression &
#CUDA_VISIBLE_DEVICES=2 python $training_script -lr 4e-4 -w 1.0,14.0 --num_classes 1 --dataset wsc --label dep --task regression &

## combined binary antidepressant w/ attention layer (lr)
#CUDA_VISIBLE_DEVICES=0 python $training_script -lr 1e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary --attention --add_name _att #&
# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 2e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary --attention --add_name _att &
# CUDA_VISIBLE_DEVICES=2 python $training_script -lr 4e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary --attention --add_name _att &
# CUDA_VISIBLE_DEVICES=3 python $training_script -lr 8e-5 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary --attention --add_name _att &

## combined binary antidepressant w/ attention layer (layer sizes)
# CUDA_VISIBLE_DEVICES=4 python $training_script -lr 4e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary --layers 256,128,16 --attention --add_name _att_256,128,16 &
# CUDA_VISIBLE_DEVICES=5 python $training_script -lr 4e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary --layers 512,128,16 --attention --add_name _att_512,128,16 &
# CUDA_VISIBLE_DEVICES=6 python $training_script -lr 4e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary --layers 128,64,16 --attention --add_name _att_128,64,16 &
# CUDA_VISIBLE_DEVICES=7 python $training_script -lr 4e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2_wsc --label antidep-binary --layers 256,64,16 --attention --add_name _att_256,64,16 &

## binary happy/sad w/ 44 threshold
#CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-4 -w 1.0,14.0 --num_classes 2 --dataset wsc --label dep-binary --add_name _thr_44

## binary happy/sad w/36 threshold, only control people
#CUDA_VISIBLE_DEVICES=0 python $training_script -lr 1e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep-binary --control --add_name _control_th_36_no_att
#CUDA_VISIBLE_DEVICES=1 python $training_script -lr 2e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep-binary --control --add_name _control_th_36_no_att #&
#CUDA_VISIBLE_DEVICES=2 python $training_script -lr 4e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep-binary --control --add_name _control_th_36_no_att &

## testing k-fold stuff
#CUDA_VISIBLE_DEVICES=0 python $training_script -lr 1e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label antidep --num_epochs 50

## multi-head attention hyperparam tuning
# CUDA_VISIBLE_DEVICES=0 python $training_script -lr 4e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep --layer 256,64,16 --num_heads 3 -bs 16 --dropout 0 --control --add_name _dropout0_nofc0 &
# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 4e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep --layer 256,64,16 --num_heads 3 -bs 16 --batch_norms 0,0,0 --control --add_name _bn00_nofc0 &
# CUDA_VISIBLE_DEVICES=2 python $training_script -lr 4e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep --layer 256,64,16 --num_heads 3 -bs 16 --batch_norms 0,1,0 --control --add_name _bn10_nofc0 &
# CUDA_VISIBLE_DEVICES=3 python $training_script -lr 4e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep --layer 256,64,16 --num_heads 3 -bs 16  --batch_norms 0,0,1 --control --add_name _bn01_nofc0 &
# CUDA_VISIBLE_DEVICES=0 python $training_script -lr 4e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep --layer 256,64,16 --num_heads 3 -bs 16  --batch_norms 0,1,1 --control --add_name _bn11_nofc0 &
# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 4e-4 -w 1.0,10.0 --num_classes 2 --dataset wsc --label dep --layer 256,64,16 --num_heads 3 -bs 16 --dropout 0.5 --control --add_name _dropout0.5_nofc0 &

## retrain shhs2 antidep predictor (lr 0.001)
# CUDA_VISIBLE_DEVICES=0 python $training_script -lr 1e-3 -w 1.0,14.0 --num_classes 2 --dataset shhs2 --label benzo --no_attention &
# CUDA_VISIBLE_DEVICES=1 python $training_script -lr 1e-4 -w 1.0,14.0 --num_classes 2 --dataset shhs2 --label benzo --no_attention &

## test run to make sure best model saves on each fold

## positive weight ablation w/new simon_model
# CUDA_VISIBLE_DEVICES=0 python3 $training_script -lr 4e-4 -w 1.0,2.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# CUDA_VISIBLE_DEVICES=1 python3 $training_script -lr 4e-4 -w 1.0,4.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# CUDA_VISIBLE_DEVICES=2 python3 $training_script -lr 4e-4 -w 1.0,6.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# CUDA_VISIBLE_DEVICES=3 python3 $training_script -lr 4e-4 -w 1.0,8.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# wait
# CUDA_VISIBLE_DEVICES=0 python3 $training_script -lr 4e-4 -w 1.0,10.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# CUDA_VISIBLE_DEVICES=1 python3 $training_script -lr 4e-4 -w 1.0,12.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# CUDA_VISIBLE_DEVICES=2 python3 $training_script -lr 4e-4 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# CUDA_VISIBLE_DEVICES=3 python3 $training_script -lr 4e-4 -w 1.0,16.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# wait
# CUDA_VISIBLE_DEVICES=0 python3 $training_script -lr 4e-4 -w 1.0,18.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &
# CUDA_VISIBLE_DEVICES=1 python3 $training_script -lr 4e-4 -w 1.0,20.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _simonmodelweight2 --control &

#CUDA_VISIBLE_DEVICES=0 python3 $training_script -lr 1e-4 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 3 --dataset shhs2 --label antidep --num_epochs 50 --simon_model --add_name simonmodelv2 #&
#CUDA_VISIBLE_DEVICES=1 python3 $training_script -lr 2e-4 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 3 --dataset shhs2 --label antidep --num_epochs 50 --simon_model --add_name simonmodelv2 &
#CUDA_VISIBLE_DEVICES=2 python3 $training_script -lr 8e-5 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 3 --dataset shhs2 --label antidep --num_epochs 50 --simon_model --add_name simonmodelv2 &

#CUDA_VISIBLE_DEVICES=0 python3 $training_script -lr 1e-4 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 3 --dataset shhs2 --label antidep --num_epochs 50 --simon_model --add_name simonmodel &
#CUDA_VISIBLE_DEVICES=1 python3 $training_script -lr 1e-4 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 3 --dataset wsc --label antidep --num_epochs 50 --simon_model --add_name simonmodel &

CUDA_VISIBLE_DEVICES=0 python3 $training_script -lr 2e-4 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --add_name _newmetricworks? --control

wait
echo "All trainings complete!"