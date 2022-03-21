CUDA_VISIBLE_DEVICES=0 python train.py \
                        --num_workers 4 --epochs 50 --model resnet50 \
                        --optim adam --lr 1e-3 \
                        --log_path ./log/ 

CUDA_VISIBLE_DEVICES=1 python train.py \
                        --num_workers 4 \
                        --epochs 50 \
						--model resnet34 \
                        --log_path ./log/ 
                        
kaggle competitions submit -c 11775-s22-hw1 -f filename.csv -m "test"