CUDA_VISIBLE_DEVICES=0 python train.py \
                        --num_workers 4 --epochs 50 --model resnet34 --optim adam --lr 1e-3 \
                        --audio_path ./ckpt/audio_resnet34.pth --video_path ./ckpt/video_mlp.pth \
                        --log_path ./log/
