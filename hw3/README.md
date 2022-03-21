
# HW3: Multimodel Fusion for MED

## Install Dependencies

```bash
conda create -n lsma-hw3 python=3.9
conda activate lsma-hw3
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Generate MFCC

* Run `audio_preprocess.py`

    ```bash
    python audio_preprocess.py \
            -i data/wav/ \
            -o data/mfcc/
    ```