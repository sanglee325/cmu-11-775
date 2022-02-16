# code for training mlp model
python train_mlp.py \
        ~/hdd/cmu/data/hw1/bof/ \
        50 \
        ~/hdd/cmu/data/hw1/labels/train_val.csv \
        models/mfcc-50.mlp.model

# code for testing trained mlp model
python test_mlp.py \
        models/mfcc-50.mlp.model \
        ~/hdd/cmu/data/hw1/bof \
        50 \
        ~/hdd/cmu/data/hw1/labels/test_for_students.csv \
        results/mfcc-50.mlp.csv