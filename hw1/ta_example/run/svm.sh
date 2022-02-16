# code for training svm model
python train_svm_multiclass.py \
        ~/hdd/cmu/data/hw1/bof/ \
        50 \
        ~/hdd/cmu/data/hw1/labels/train_val.csv \
        models/mfcc-50.svm.multiclass.model

# run test    
python test_svm_multiclass.py \
        models/mfcc-50.svm.multiclass.model \
        ~/hdd/cmu/data/hw1/bof \
        50 \
        ~/hdd/cmu/data/hw1/labels/test_for_students.csv \
        results/mfcc-50.svm.multiclass.csv