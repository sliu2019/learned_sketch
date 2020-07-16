# Learned sketches for low-rank approx and regression

### Prerequisites

Conda env containing all necessary Python packages has been included (learn_sketch_env.yml)
Download data at links provided in paper

### Main files

train_regression.py : runs SGD algorithm on Multiple Response Regression (MRR) task
regression_init_utils.py: runs greedy algorithm for MRR

train_lp_regression: runs SGD algorithm for Lp regression
train_huber_regression: runs SGD algorithm for Huber regression

train_speedup_direct.py : runs SGD algorithm on Low-Rank Approximation (LRA) task using the one-sketch alg
train_4sketch.py: runs SGD algorithm for LRA using the four-sketch alg
sparsity_pattern_init_algs.py: the function init_w_greedy_gs(...) runs greedy algorithm for LRA
run_LRA_baselines.py: runs "exact SVD" baseline for LRA

run_kmeans.py: evaluates all sketching algorithms for k-means clustering

### How to run
To run these algorithms (except for the last two files), pass arguments by command line. Examples of usage in sample.sh

For run_LRA_baselines.py and run_kmeans.py, edit the parameters in the main function and run the script 

A description of all arguments can be found in the "make_parser" functions of each script

Prior to running, you will have to update the paths to the data and pass argument "--raw" to process the raw data

### Additional files of interest
The folder ./data contains files for processing raw data to the matrices used in LRA or regression


