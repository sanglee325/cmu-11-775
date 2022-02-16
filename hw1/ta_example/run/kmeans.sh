# prepare data for kmeans
python select_frames.py \
        --input_path ~/hdd/cmu/data/hw1/labels/train_val.csv \
        --ratio 0.2 \
        --output_path result/selected.mfcc.csv \
        --mfcc_dir ~/hdd/cmu/data/hw1/mfcc

# train kmeans model
python train_kmeans.py \
        -i ./result/selected.mfcc.csv \
        -k 50 \
        -o models/kmeans.50.model