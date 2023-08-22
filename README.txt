this readme describes how to run all relevant files in this repo and avoid possible bugs.

for all of these files, make sure you set the args parameters accordingly, any savepaths accordingly,
and load/split the data accordingly.

--- TRAINING ---

__init__.py (in dataset folder):
contains all the dataset classes.

model.py:
contains all the models. SimonModel is most relevant atm.

trainingLoop.py:
here is an example batch script to run the SimonModel with some relevant hyperparameters

CUDA_VISIBLE_DEVICES=0 python3 trainingLoop.py -lr 4e-4 -w 1.0,14.0 -bs 16 --num_classes 2 --num_heads 4 --dataset wsc --label dep --num_epochs 30 --simon_model --control --add_name _myrun

change the folder save path to where you want the tensorboard logs to save in your scratch, and
add in stuff to save models if you want (I have previous code for that commented out if you want to use it).
note that the cli arguments all have descriptions if they're relevant to you, so read those for info on them.

metrics.py:
contains all the metrics classes, including some custom ones i made. edit as you need. some of the args parameters contribute to which metrics 
get used, so keep that in mind.

--- EVALUATION ---

eval.py:
to evaluate the SimonModel performance, run 'python3 eval.py', but in the code, manually set the model path, and also the following args beforehand:
args.control, args.tca, args.ssri, args.other

one should be true, the others false. if evaluating on wsc control (assuming you trained on it), make sure to uncomment the fold stuff and use fold 0.
the code is a bit of a mess right now and currently just generates the confusion matrix, but there's other things you can use that are commented out.

UDALL STUFF:
- udallPreds.py creates csv for each patient containing the raw probability prediction and the thresholded one. change
  the threshold to what you want
- udallPos.py is to get csv of the ratio of positive nights / total nights for each udall patient.
- udallPlot.py to plot the tsne for each patient's set of night predictions

For all of these, make sure to run the right model and change the file saving names accordingly.

--- VISUALIZATION ---

plotCDF.py

plotROC.py:
run 'python3 plotROC.py'. manually set the model you want to use in the code, as well as the figure save path. also,
if you want to generate the prc instead then just change the plot_auroc to plot_auprc at the end of the code.

tsne_visualization.py:
NOTE - for the tsne coloring to work, you need to change the dataset class (__init__.py) so that the data_dict created is
the same as the data_dict for the label you want to color by. for example, right now if the label is either 'antidep' or 'nsrrid',
the data dict will be the same for both and the tsne will work in that case. if your label is benzos, however, then 
you need to move the "or label=='nsrrid'" to the benzo if statement.

tsne_visualization_udall.py:

plotThreshMetrics.py:

--- BASELINES ---

svm.py, randomForest.py, decisionTree.py, and logisticRegression.py are all very similar. just comment/uncomment which datasets you want to use.
you can also alter the hyperparameter grids for each of them.


NOTE: BrEEG folder is Hao stuff