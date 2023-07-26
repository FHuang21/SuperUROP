here is my readme for training my model and evaluating/visualizing its performance

TRAIN:
here is an example batch script to run the SimonModel with some relevant hyperparameters

CUDA_VISIBLE_DEVICES=0 python3 trainingLoop.py -lr 4e-4 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 50 --simon_model --control --add_name _bruh

TO-DO: change the folder save path to where you want the tensorboard logs to save in your scratch.
note: the cli arguments all have descriptions if they're relevant to you, so read those for info on them.
note 2: i set data source as eeg even though it's technically mage breathing. you can change it and it will only change the experiment name.

EVALUATE:
to evaluate the SimonModel performance, run 'python3 eval.py', but in the code, manually set the model path, and also the following args beforehand:
args.control, args.tca, args.ssri, args.other

one should be true, the others false. if evaluating on wsc control (assuming you trained on it), make sure to uncomment the fold stuff and use fold 0.

ROC CURVE:
run 'python3 plotROC.py'. again, manually set the model you want to use in the code, as well as the figure save path.