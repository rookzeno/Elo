Hello!

Below you can find a outline of how to reproduce my solution for the <Elo Merchant Category Recommendation> competition.

If you run into any trouble with the setup/code or have any questions please contact me at <email>

# HARDWARE: (The following specs were used to create the original solution)
Ubuntu 16.04 LTS (512 GB boot disk)
DDR4-RAM 64GB
Intel CPU Core i7-8700K
GeForce GTX 1080 Ti *1

# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.5.1
CUDA 10.0
cuddn 7.3.0
nvidia drivers v.410.79

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
nothing.

# DATA PROCESSING
python ./python prepare_data.py 

# MODEL BUILD: There are two options to produce the solution.
1) very fast prediction<br>
    a) runs in a 5 hours<br>
    b) uses lightgbm predictions
2) ordinary prediction<br>
    a) expect this to run for 1-2 days<br>
    b) uses stacking model files<br>

shell command to run each build is below

1) very fast prediction (overwrites comp_preds/sub1.csv and comp_preds/sub2.csv)
python ./python prepare_data.py
python ./python predict_single_model.py


2) ordinary prediction (overwrites predictions in comp_preds directory)
python ./python prepare_data.py
python ./python train.py
python ./python test.py


