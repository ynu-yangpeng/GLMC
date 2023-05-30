# 🌎[CVPR2023] Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions（GLMC）
by **Fei Du, Peng Yang, Qi Jia, Fengtao Nan, Xiaoting Chen, Yun Yang**

This is the official implementation of  [Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions](https://arxiv.org/abs/2305.08661)

🎬[Video](https://www.youtube.com/watch?v=mTeWItl4k9k) | 💻[Slide](https://drive.google.com/file/d/151-tHX2CsuOGmNFHsTxkancZBvf4ZkSB/view?usp=share_link) | 🔥[Poster](https://drive.google.com/file/d/1wEWKrDLBiZAjVf3E7OFvhIi0wg4ME1Tj/view?usp=share_link)

## Update 2023/5/23

>Thank you very much for the question from @[CxC-ssjg](https://github.com/CxC-ssjg). In our code for the Cifar10Imbalance and Cifar100Imbalance classes, when generating imbalanced data, we used np.random.choice for random sampling of samples. However, we did not set the "replace" parameter in the method to False, which could result in multiple repeated samples of a particular sample, thereby reducing the diversity of the dataset. Based on @[CxC-ssjg](https://github.com/CxC-ssjg)'s advice, we set replace to False and fine-tuned our model accordingly. As a result, we observed a significant improvement in performance compared to the results reported in the paper. We have provided an update on the latest results and made the model publicly available. Once again, thank you, @[CxC-ssjg](https://github.com/CxC-ssjg), for your valuable question.

| Dateset | IF | GLMC | GLMC(Updated) | GLMC(Updated) + MaxNorm |
| :---: |:---:|:---:|:---:|:---:|
| CIFAR-100-LT | 100   | 55.88%    | [57.97%](https://drive.google.com/file/d/1QNusy82tFNCK_Urz7cdztpHRV-9gQoW3/view?usp=share_link) | 58.41%    |
| CIFAR-100-LT | 50    | 61.08%    | [63.78%](https://drive.google.com/file/d/1QEoDpwIDnK57vK-DPU6wjIYTxoG3dj98/view?usp=share_link) | 64.57%    |
| CIFAR-100-LT | 10    | 70.74%    | [73.40%](https://drive.google.com/file/d/1ZOqNwi83dW4Rj3lsEE_xuVFQMYAKrhjc/view?usp=share_link) | 74.28%    |
| CIFAR-10-LT | 100   | 87.75%    | [88.50%](https://drive.google.com/file/d/1doBLTL9-Y1Sv_2BZ-OFcFlt3rL0JxqZ0/view?usp=share_link) | 89.58%    |
| CIFAR-10-LT | 50    | 90.18%    | [91.04%](https://drive.google.com/file/d/1n7ieuDZSMODeAs20kutBbEncn3RUavqe/view?usp=share_link) | 92.04%    |
| CIFAR-10-LT | 10    | 94.04%    | [94.87%](https://drive.google.com/file/d/1YYxqUR90J1ab0UKs4S2Qb09yi_9jFOjD/view?usp=share_link) | 95.00%    |





## Update 2023/5/15
> Apologies for the oversight in our paper regarding the incorrect upload of the results for CIFAR-10. We have updated our GitHub repository and reported the final results for CIFAR-10-LT.
> Compared to the latest state-of-the-art work by BCL[1], our results are still 3% higher. We have also uploaded the latest paper on arXiv, and you can find it at the following link: [Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions](https://arxiv.org/abs/2305.08661)

The experimental setup was as follows: 

````
python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4
````

### CIFAR-10-LT
| Method | IF | Model | Top-1 Acc(%) |
| :---:| :---:|:---:|:---:|
| GLMC   | 100   | ResNet-32     | 87.75%    |
| GLMC   | 50    | ResNet-32     | 90.18%    |
| GLMC   | 10    | ResNet-32     | 94.04%    |
| GLMC + MaxNorm   | 100   | ResNet-32     | 87.57%    |
| GLMC + MaxNorm   | 50    | ResNet-32     | 90.22%    |
| GLMC + MaxNorm   | 10    | ResNet-32     | 94.03%    |

[1] Jianggang Zhu, ZhengWang, Jingjing Chen, Yi-Ping Phoebe Chen, and Yu-Gang Jiang. Balanced contrastive learning for long-tailed visual recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6908–6917, 2022. 2, 3, 5, 6

### 💥Meanwhile, We supplemented the experiment on iNaturelist2018 and achieved the state-of-the-art.
| Method | Model | Many | Med | Few | All | model |
| :---:| :---:|:---:|:---:|:---:| :---:|  :---:| 
| GLMC |ResNeXt-50 | 64.60  | 73.16  | 73.01     | 72.21    | [Download](https://drive.google.com/file/d/1dcE1eJaRAtMmIF3qTzALnMzTb9AbC3mb/view?usp=share_link) |


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

for CIFAR-10-LT
````
python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1

python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1

python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.1 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.2 --label_weighting 1  --contrast_weight 2
````

for CIFAR-100-LT
````
python main.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2  --contrast_weight 4

python main.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.2  --label_weighting 1.2  --contrast_weight 6

python main.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.1 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.2  --label_weighting 1.2  --contrast_weight 4
````


for ImageNet-LT
````
python main.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.1 --epochs 135 -b 120 --momentum 0.9 --weight_decay 2e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
````

for iNaturelist2018 
````
python main.py --dataset iNaturelist2018 -a resnext50_32x4d --num_classes 8142 --beta 0.5 --lr 0.1 --epochs 120 -b 128 --momentum 0.9 --weight_decay 1e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
````

## Testing
````
python test.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --resume model_path
````

## Result and Pretrained models

### CIFAR-10-LT
| Method | IF | Model | Top-1 Acc(%) |
| :---:| :---:|:---:|:---:|
| GLMC   | 100   | ResNet-32     | 87.75%    |
| GLMC   | 50    | ResNet-32     | 90.18%    |
| GLMC   | 10    | ResNet-32     | 94.04%    |
| GLMC + MaxNorm   | 100   | ResNet-32     | 87.57%    |
| GLMC + MaxNorm   | 50    | ResNet-32     | 90.22%    |
| GLMC + MaxNorm   | 10    | ResNet-32     | 94.03%    |

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

### iNaturelist2018     
| Method | Model | Many | Med | Few | All | model |
| :---:| :---:|:---:|:---:|:---:| :---:|  :---:| 
| GLMC |ResNeXt-50 | 64.60  | 73.16  | 73.01     | 72.21    | [Download](https://drive.google.com/file/d/1dcE1eJaRAtMmIF3qTzALnMzTb9AbC3mb/view?usp=share_link) |


## Citation
If you find this code useful for your research, please consider citing our paper<br>
````
@inproceedings{
du2023global,
title={Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions},
author={Fei Du, Peng Yang, Qi Jia, Fengtao Nan, Xiaoting Chen, Yun Yang},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023},
url={https://arxiv.org/abs/2305.08661}
}
````
