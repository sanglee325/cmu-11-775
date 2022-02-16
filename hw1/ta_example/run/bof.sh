# create bag of features
python get_bof.py \
        models/kmeans.50.model \
        50 videos.name.lst \
        --mfcc_path ~/hdd/cmu/data/hw1/mfcc/ \
        --output_path ~/hdd/cmu/data/hw1/bof/