# debug for SIFT generation
python code/run_sift.py data/labels/debug.csv --debug

# generate SIFT for train_val data
python code/run_sift.py data/labels/train_val.csv

# generate SIFT for test data
python code/run_sift.py data/labels/test_for_students.csv


# debug k_means with train
python code/train_kmeans.py data/labels/debug.csv data/sift/ 128 sift_128 --debug

# train k_means with train
python code/train_kmeans.py data/labels/train_val.csv data/sift/ 128 sift_128


# debug bow with train
python code/run_bow.py data/labels/debug.csv sift_128 data/sift --debug

# get bow with train_val
python code/run_bow.py data/labels/train_val.csv sift_128 data/sift
python code/run_bow.py data/labels/test_for_students.csv sift_128 data/sift


# extract CNN features
python code/run_cnn.py data/labels/debug.csv --debug

# get CNN features
python code/run_cnn.py data/labels/train_val.csv
python code/run_cnn.py data/labels/test_for_students.csv

python code/run_cnn.py data/labels/train.csv
python code/run_cnn.py data/labels/eval.csv

# bow mlp
python code/run_mlp.py sift --feature_dir data/bow_sift_128 --num_features 128
python code/run_mlp.py cnn --feature_dir data/cnn-eval --num_features 512