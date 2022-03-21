# HW3: Multimodel Fusion for MED

## Install Dependencies

```bash
conda create -n lsma-hw3 python=3.9
conda activate lsma-hw3
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Audio Feature Training

* Generate MFCC features. Run `audio_preprocess.py`

    ```bash
    cd audio
    python audio_preprocess.py \
            -i data/wav/ \
            -o data/mfcc/
    ```

* Train neural network. Run `train.py`.

    ```bash
    cd audio
    CUDA_VISIBLE_DEVICES=0 python train.py \
                        --num_workers 4 --epochs 50 --model resnet34 \
                        --optim adam --lr 1e-3 \
                        --log_path ./log/
    ```

## Video Feature Training

* Generate CNN features. (Follow the guide of hw2)

* Train neural network. Run `train.py`.

    ```bash
    cd video
    CUDA_VISIBLE_DEVICES=0 python train.py \
                        --num_workers 4 --epochs 50 --model MLP \
                        --optim adam --lr 1e-3 \
                        --log_path ./log/ 
    ```

## Multimodel Fusion

* Extract features from to models above. Concat the features from the models and train the classifier.

    ```bash
    cd fusion
    CUDA_VISIBLE_DEVICES=0 python train.py \
                        --num_workers 4 --epochs 50 --model resnet34 --optim adam --lr 1e-3 \
                        --audio_path <path_to_audio.pth> --video_path <path_to_video.pth> \
                        --log_path ./log/
    ```


