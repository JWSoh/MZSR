# Large-Scale Training Codes

## Environments
- Ubuntu 18.04
- [Tensorflow 1.8](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.1
- Python 3.6

## Guidelines for Codes

**Requisites should be installed beforehand.**

### Training

Download training dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

#### Generate TFRecord dataset
- Refer to [MainSR](https://www.github.com/JWSoh/MainSR) repo.

#### Train "bicubic" pretrained model

** An Example Code **
```
python main.py --gpu 0 --trial 1 --step 0
```

[Options]
```
python main.py --gpu [GPU_number] --trial [Trial of your training] --step [Global step]

--gpu: If you have more than one gpu in your computer, the number denotes the index. [Default 0]
--trial: Trial number. Any integer numbers can be used. [Default 0]
--step: Global step. When you resume the training, you need to specify the right global step. [Default 0]
```