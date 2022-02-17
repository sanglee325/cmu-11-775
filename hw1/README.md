# HW1: Audio-based Multimedia Event Detection

## Get Audio Features

### MFCCs

* Install FFMPEG.

    ```bash
    sudo apt install ffmpeg
    ```

* Extract audio from videos.

    ```bash
    for file in videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -f wav wav/${filename}.wav; done
    ```

* Create conda environment.

    ```bash
    conda create -n audio-med python=3.8
    conda activate audio-med
    ```

* Install requirements.txt.

    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```

* Run `extract_mfcc.py`.

    ```bash
    python extract_mfcc.py \
            -i wav/ \
            -o mfcc/
    ```
* Spilt data into train and validation set. Run `split_data.py`.

    ```bash
    python split_data.py \
            --input_file labels/trainval.csv \
            --ratio 0.1 \
            --output_path labels/
    ```

* K-means Clustering

    ```bash
    # generate mfcc file from seperated training data
    python select_frames.py \
            --input_path labels/train.csv \
            --ratio 1 \
            --output_path mfcc/train.mfcc.csv \
            --mfcc_dir mfcc/

    # train kmeans model
    python train_kmeans.py \
            -i mfcc/train.mfcc.csv \
            -k 50 \
            -o models/kmeans.50.model
    ```

* Feature extraction

    ```bash
    # get list of videos
    ls videos/ | while read line;do filename=$(basename $line .mp4);echo $filename;done > videos.name.lst

    # extract bag of feature for each video
    python get_bof.py \
            models/kmeans.50.model \
            50 \
            videos.name.lst \
            --mfcc_path mfcc40/ \
            --output_path bof/
    ```

### SoundNet

* Follow instructions on [soundnet_pytorch](https://github.com/salmedina/soundnet_pytorch) to extract audio features.

## Train MLP model

* Train MLP model. Run `train.py`.

    ```bash
    python train.py \
            <feature_dir> \
            <feature_dim> \
            labels/train.csv \
            <model_path>
    ```

* Test and generate result file. Run `test.py`

    ```bash
    python test.py \
            <model_path> \
            <feature_dir> \
            <feature_dim> \
            labels/validation.csv \
            <result_file>
    ```