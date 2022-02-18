# CMU 11-775 Spring 2022 Homework 2

[PDF Handout](docs/handout.pdf)

In this homework we will perform a video classification task with visual features.

## Recommended Hardware

This code template is built based on [PyTorch](https://pytorch.org) and [Pyturbo](https://github.com/CMU-INF-DIVA/pyturbo) to fully utilize the computation of multiple CPU cores and GPUs.
SIFT feature, K-Means, and Bag-of-Words must run on CPUs, while CNN features and MLP classifiers can run on GPUs.
For AWS, a `g4dn.4xlarge` instance should be sufficient for the full pipeline.
During initial debugging, you are recommended to use a smaller instance to save money, e.g., `g4dn.xlarge` or a CPU-only instance for the SIFT part.
For more about AWS, see this [Doc](https://docs.google.com/document/d/1XkpGSzInT5TJz0hc0jUd7j5kGvuGO_wTOATW8pp4GCg/edit?usp=sharing) (Andrew ID required).

## Install Dependencies

A set of dependencies is listed in [environment.yml](environment.yml). You can use `conda` to create and activate the environment easily.

```bash
conda env create -f environment.yml
conda activate 11775-hw2
```

Optionally, you can create the environment in a specified path (e.g., your EFS directory)

```bash
conda env create -f environment.yml -p <path>
conda activate <path>
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

## Development and Debugging

Some functions in the pipeline are deliberately left blank for you to implement, where an `NotImplementedError` will be raised.
We recommend you generate a small file list (e.g. `debug.csv` with 20 lines) for fast debugging during initial development.
The `--debug` option in some scripts are also very helpful.
In addition, you can enable `pdb` debugger upon exception

```bash
# Instead of 
python xxx.py yyy zzz
# Run
ipython --pdb xxx.py -- yyy zzz
```

## SIFT Features

To extract SIFT features, use

```bash
python code/run_sift.py data/labels/xxx.csv
```

By default, features are stored under `data/sift`. As an estimate for you, it took around 1 hour to run the `train_val` set on a server with 10 hyperthreaded CPU cores when we only select every 20 frames and extract 32 key points from each.

To train K-Means with SIFT feature for 128 clusters, use

```bash
python code/train_kmeans.py data/labels/xxx.csv data/sift 128 sift_128
```

By default, model weights are stored under `data/kmeans`. With 10% of the feature vectors, it took less than 5 minutes to train. You can use more data for a potentially better performance but longer training time.

To extract Bag-of-Words representation with the trained model, use

```bash
python code/run_bow.py data/labels/xxx.csv sift_128 data/sift
```

By default, features are stored under `data/bow_<model_name>` (e.g., `data/bow_sift_128`).

## CNN Features

To extract CNN features, use

```bash
python code/run_cnn.py data/labels/xxx.csv
```

By default, features are stored under `data/cnn`.

The current pipeline processes images one by one, which is not so friendly with GPU.
You can try to optimize it into batch processing for faster speed.

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

By default, training logs and predictions are stored under `data/mlp/cnn/version_xxx/`.
You can directly submit the CSV file to [Kaggle](https://www.kaggle.com/c/11775-s22-hw1-p1/overview).
