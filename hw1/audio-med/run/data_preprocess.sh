# split files into train and validation set
python split_data.py \
        --input_file ../../data/labels/train_val.csv \
        --ratio 0.1 \
        --output_path ../../data/labels

# save into train and validation
python split_file.py \
        --input_path ../../data/mfcc \
        --output_path ../../data/