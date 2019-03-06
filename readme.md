Hello!

Below you can find a outline of how to reproduce my solution for the <Elo Merchant Category Recommendation> competition.
If you run into any trouble with the setup/code or have any questions please contact me at <email>

#ARCHIVE CONTENTS
kaggle_model.tgz          : original kaggle model upload - contains original code, additional training examples, corrected labels, etc
comp_etc                     : contains ancillary information for prediction - clustering of training/test examples
comp_mdl                     : model binaries used in generating solution
comp_preds                   : model predictions
train_code                  : code to rebuild models from scratch
predict_code                : code to generate predictions from model binaries

#HARDWARE: (The following specs were used to create the original solution)
Ubuntu 16.04 LTS (512 GB boot disk)
DDR4-RAM 64GB
Intel CPU Core i7-8700K
GeForce GTX 1080 Ti *1

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.5.1
CUDA 10.0
cuddn 7.3.0
nvidia drivers v.410.79

#DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
# below are the shell commands used in each step, as run from the top level directory
nothing.

#DATA PROCESSING
# The train/predict code will also call this script if it has not already been run on the relevant data.
python ./python prepare_data.py 

#MODEL BUILD: There are two options to produce the solution.
1) very fast prediction
    a) runs in a 5 hours
    b) uses lightgbm predictions
2) ordinary prediction
    a) expect this to run for 1-2 days
    b) uses stacking model files

shell command to run each build is below
#1) very fast prediction (overwrites comp_preds/sub1.csv and comp_preds/sub2.csv)
python ./python prepare_data.py
python ./python predict_single_model.py


#2) ordinary prediction (overwrites predictions in comp_preds directory)
python ./python prepare_data.py
python ./python train.py
python ./python test.py


