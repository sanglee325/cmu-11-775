# HW2P1: Video-based MED

## Install Dependencies

A set of dependencies is listed in [environment.yml](environment.yml). You can use `conda` to create and activate the environment easily.

```bash
conda env create -f environment.yml
conda activate 11775-hw2
```

## Dataset

You will be using two parts of data for this homework:

* Data from [Homework 1](https://github.com/11775website/11775-hws/tree/master/spring2022/hw1#data-and-labels) which you should have downloaded. [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip).
* A new larger set of test videos. [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data_p2.zip).

Both parts should be decompressed under the `data` directory.
You can directly download them into your AWS virtual machine:

```bash
mkdir data && cd data
# Download and decompress part 1 data (no need if you still have it from HW1)
wget https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip
unzip 11775_s22_data.zip
# Download and decompress part 2 data
wget https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data_p2.zip
unzip 11775_s22_data_p2.zip
```

## SIFT Features

To extract SIFT features, use

```bash
python code/run_sift.py data/labels/train_val.csv
python code/run_sift.py data/labels/test_for_students.csv
```

To train K-Means with SIFT feature for 128 clusters, use

```bash
python code/train_kmeans.py data/labels/train_val.csv data/sift/ 128 sift_128
```

To extract Bag-of-Words representation with the trained model, use

```bash
python code/run_bow.py data/labels/train_val.csv sift_128 data/sift
python code/run_bow.py data/labels/test_for_students.csv sift_128 data/sift
```

By default, features are stored under `data/bow_<model_name>` (e.g., `data/bow_sift_128`).

## CNN Features

To extract CNN features, use

```bash
python code/run_cnn.py data/labels/train_val.csv
python code/run_cnn.py data/labels/test_for_students.csv
```

By default, features are stored under `data/cnn`.

## MLP Classifier

The training script automatically and deterministically split the `train_val` data into training and validation, so you do not need to worry about it.

To train MLP with SIFT Bag-of-Words, run

```bash
python code/run_mlp.py sift --feature_dir data/bow_sift_128 --num_features 128
```

To train MLP with CNN features, run

```bash
python code/run_mlp.py cnn --feature_dir data/cnn --num_features <num_feat>
```

