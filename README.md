# Multimodal Information Modulation
This is the open source code for paper: Multimodal Reaction: Information Modulation for Cross-modal Representation Learning

## Table of Contents
- [Description](##Description)
- [Preparation](##Preparation)
- [Running](##Running)

## Description
This is the open source code for paper: Multimodal Reaction: Information Modulation for Cross-modal Representation Learning. We have provided the implementation on the task of multimodal sentiment analysis. The main components are as follow:
1. `./datasets` contains the datasets used in the experiments
2. `./modules` contains the model definition
3. `./utils` contains the functions for data processing, evaluation metrics, etc.
4. `global_configs.py` defines important constants
5. `train.py` defines the training process

## Preparation
### Datasets
We have already provided the processed MOSI dataset in `./dataset`

To download the larger MOSEI dataset, you can run `datasets/download_datasets.sh`

### Installation
To install the required packages, you can run `pip install -r requirements.txt`

### Configuration
Before starting training, you should define the global constants in `global_configs.py`. Default configuration is set to MOSI dataset. Important settings include GPU setting, learning rate, feature dimension, training epochs, training batch size, dataset setting and dimension setting of input data. To run the MOSEI dataset, remember to change dimension of the visual modality

```
from torch import nn

class DefaultConfigs(object):

    device = '1'                                 #GPU setting
    logs = './logs/'
    
    max_seq_length = 50 
    lr = 1e-5                                    #learning rate
    d_l = 80                                     #feature dimension
    n_epochs = 100                               #training epochs
    train_batch_size = 16                        #training batch size
    dev_batch_size = 128
    test_batch_size = 128
    model_choice = 'bert-base-uncased'

    dataset = 'mosi'                             #dataset setting
    TEXT_DIM = 768                               #dimension setting
    ACOUSTIC_DIM = 74
    VISUAL_DIM = 47

    # dataset = 'mosei'
    # ACOUSTIC_DIM = 74
    # VISUAL_DIM = 35
    # TEXT_DIM = 768

config = DefaultConfigs()
```

## Running
To run the experiments, you can run the following command
```
python train.py
```
Note that the proposed method is flexible to work with various fusion strategies. We have performed experiments on element-wise addition, element-wise multiplication, concatenation and [tensor fusion](https://github.com/Justin1904/TensorFusionNetworks). If tensor fusion is adopted, the setting of feature dimension `d_l` in `global_configs.py` can be larger to ensure higher capacity.

## Acknowledgments
We would like to express our gratitude to [huggingface](https://huggingface.co/) and [MAG-BERT](https://github.com/WasifurRahman/BERT_multimodal_transformer), which are of great help to our work.
