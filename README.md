# Tensorflow-DenseNet with ImageNet Pretrained Models

This is an [Tensorflow](https://www.tensorflow.org/) implementation of [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) by G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten with [ImageNet](http://www.image-net.org/) pretrained models. The weights are converted from [DenseNet-Keras Models](https://github.com/flyyufelix/DenseNet-Keras).

The code are largely borrowed from [TensorFlow-Slim Models](https://github.com/tensorflow/models/tree/master/slim).

## Pre-trained Models

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN)

Network|Top-1|Top-5|Checkpoints
:---:|:---:|:---:|:---:
DenseNet 121 (k=32)| 74.91| 92.19| [model](https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA)
DenseNet 169 (k=32)| 76.09| 93.14| [model](https://drive.google.com/open?id=0B_fUSpodN0t0TDB5Ti1PeTZMM2c)
DenseNet 161 (k=48)| 77.64| 93.79| [model](https://drive.google.com/open?id=0B_fUSpodN0t0NmZvTnZZa2plaHc)

## Usage
Follow the instruction [TensorFlow-Slim Models](https://github.com/tensorflow/models/tree/master/slim).

### Step-by-step Example of training on flowers dataset.
#### Downloading ans converting flowers dataset

```
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

#### Training a model from scratch.

```
$ DATASET_DIR=/tmp/data/flowers
$ TRAIN_DIR=/tmp/train_logs
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=densenet121 
```

#### Fine-tuning a model from an existing checkpoint

```
$ DATASET_DIR=/tmp/data/flowers
$ TRAIN_DIR=/tmp/train_logs
$ CHECKPOINT_PATH=/tmp/my_checkpoints/tf-densenet121.ckpt
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=densenet121 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=global_step,densenet121/logits \
    --trainable_scopes=densenet121/logits
```
