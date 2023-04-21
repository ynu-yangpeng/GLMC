# [CVPR2023] Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions（GLMC）
by **Fei Du, Peng Yang, Qi Jia, Fengtao Nan, Xiaoting Chen, Yun Yang**

This is the official implementation of  [Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions](https://openreview.net/forum?id=gLkKCqK3WOD)



## Overview

<div align="center"><img src="https://user-images.githubusercontent.com/48430480/223947913-edbdd463-d6e1-4ae7-8e8d-b846c002a20d.png"></div>


> An overview of our GLMC: two types of mixed-label augmented images are processed by an encoder network and a projection head to obtain the representation $h_g$ and $h_l$. Then a prediction head transforms the two representations to output $u_g$ and $u_l$. We minimize their negative cosine similarity as an auxiliary loss in the supervised loss. $sg(*)$ denotes stop gradient operation.

> 
![image](https://user-images.githubusercontent.com/48430480/222028170-e63da465-e143-4c6d-bdb9-ca1b3e31d469.png)


> We propose an efficient one-stage training strategy for long-tailed visual recognition called Global and Local Mixture Consistency cumulative learning (GLMC). Our core ideas are twofold: (1) a global and local mixture consistency loss improves the robustness of the feature extractor. Specifically, we generate two augmented batches by the global MixUp and local CutMix from the same batch data, respectively, and then use cosine similarity to minimize the difference. (2) A cumulative head-tail soft label reweighted loss mitigates the head class bias problem. We use empirical class frequencies to reweight the mixed label of the head-tail class for long-tailed data and then balance the conventional loss and the rebalanced loss with a coefficient accumulated by epochs.

## Getting Started
### Requirements
All codes are written by Python 3.9 with

- PyTorch = 1.10.0 

- torchvision = 0.11.1

- numpy = 1.22.0

### Preparing Datasets
Download the datasets CIFAR-10, CIFAR-100, ImageNet, and iNaturalist18 to GLMC-2023/data. The directory should look like

````
GLMC-2023/data
├── CIFAR-100-python
├── CIFAR-10-batches-py
├── ImageNet
|   └── train
|   └── val
├── train_val2018
└── data_txt
    └── ImageNet_LT_val.txt
    └── ImageNet_LT_train.txt
    └── iNaturalist18_train.txt
    └── iNaturalist18_val.txt
    
````
## Training
for CIFAR-100-LT
````
python main.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3
--resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 10
````


for ImageNet-LT
````
python main.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.1 --epochs 135 -b 128 --momentum 0.9 --weight_decay 2e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
````

## Testing
````
python test.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --resume model_path
````

## Result and Pretrained models

### CIFAR-10-LT
| Method | IF | Model | Top-1 Acc(%) |
| :---:| :---:|:---:|:---:|
| GLMC   | 100   | ResNet-32     | 92.34    |
| GLMC   | 50    | ResNet-32     | 94.18    |
| GLMC   | 10    | ResNet-32     | 94.92    |
| GLMC +  MaxNorm  | 100   | ResNet-32     | 94.18    |
| GLMC +  MaxNorm  | 50    | ResNet-32     | 95.13    |
| GLMC +  MaxNorm  | 10    | ResNet-32     | 95.70    |

### CIFAR-100-LT     
| Method | IF | Model | Top-1 Acc(%) |
| :---:| :---:|:---:|:---:|    
| GLMC   | 100   | ResNet-32     | 55.88    |
| GLMC   | 50    | ResNet-32     | 61.08    |
| GLMC   | 10    | ResNet-32     | 70.74    |
| GLMC +  MaxNorm  | 100   | ResNet-32     | 57.11    |
| GLMC +  MaxNorm  | 50    | ResNet-32     | 62.32    |
| GLMC +  MaxNorm  | 10    | ResNet-32     | 72.33    |

### ImageNet-LT     
| Method | Model | Many | Med | Few | All | model |
| :---:| :---:|:---:|:---:|:---:| :---:|  :---:| 
| GLMC |ResNeXt-50 | 70.1  | 52.4  | 30.4     | 56.3    | [Download](https://drive.google.com/file/d/1om0ZRuC0PYrYHA1mAsdxm31RYQ_sqDUc/view?usp=share_link) |
| GLMC + BS |ResNeXt-50 | 64.76 | 55.67    | 42.19    | 57.21   | [Download](https://drive.google.com/file/d/1GILBAR5fPcpICtM6uUwmYGkEN11wyEOV/view?usp=share_link) |


## Citation
If you find this code useful for your research, please consider citing our paper<br>
````
@inproceedings{
du2023global,
title={Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions},
author={Fei Du, Peng Yang, Qi Jia, Fengtao Nan, Xiaoting Chen, Yun Yang},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023},
url={https://openreview.net/forum?id=gLkKCqK3WOD}
}
````
