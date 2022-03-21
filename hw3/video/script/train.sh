
CUDA_VISIBLE_DEVICES=1 python train.py \
                        --num_workers 4 --epochs 50 --model MLP \
                        --optim adam --lr 1e-3 \
                        --log_path ./log/ 