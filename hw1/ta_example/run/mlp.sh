# code for training mlp model
python train_mlp.py \
        ~/hdd/cmu/data/bof/ \
        50 \
        ~/hdd/cmu/data/labels/train_val.csv \
        models/mfcc-50.mlp2.model

# code for testing trained mlp model
python test_mlp.py \
        models/mfcc-50.mlp2.model \
        ~/hdd/cmu/data/bof \
        50 \
        ~/hdd/cmu/data/labels/test_for_students.csv \
        results/mfcc-50.mlp2.csv

# code for training mlp model (soundnet)
python train_mlp.py \
        ~/hdd/cmu/data/soundnet/avgpool \
        1024 \
        ~/hdd/cmu/data/labels/train_val.csv \
        models/soundnet-1024.mlp.model

# code for testing trained mlp model (soundnet)
python test_mlp.py \
        models/soundnet-1024.mlp.model \
        ~/hdd/cmu/data/soundnet/avgpool \
        1024 \
        ~/hdd/cmu/data/labels/test_for_students.csv \
        results/soundnet-1024.mlp.csv