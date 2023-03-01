# [CVPR2023] Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions（GLMC）
by **Fei Du, peng Yang, Qi Jia, xiaoting chen, Fengtao Nan, Yun Yang**

This is the official implementation of  [Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions]

## Overview
![image](https://user-images.githubusercontent.com/48430480/222028170-e63da465-e143-4c6d-bdb9-ca1b3e31d469.png)


> We propose an efficient one-stage training strategy for long-tailed visual recognition called Global and Local Mixture Consistency cumulative learning (GLMC). Our core ideas are twofold: (1) a global and local mixture consistency loss improves the robustness of the feature extractor. Specifically, we generate two augmented batches by the global MixUp and local CutMix from the same batch data, respectively, and then use cosine similarity to minimize the difference. (2)A cumulative head-tail soft label reweighted loss mitigates the head class bias problem. We use empirical class frequencies to reweight the mixed label of the head-tail class for long-tailed data and then balance the conventional loss and the rebalanced loss with a coefficient accumulated by epochs.

## Getting Started
### Requirements
All codes are written by Python 3.9 with

- PyTorch = 1.10.0 

- torchvision = 0.11.1

- numpy = 1.22.0

### Preparing Datasets
Download the object re-ID datasets Market-1501, MSMT17, and VeRi-776 to PPLR/examples/data. The directory should look like
````
PPLR/examples/data
├── Market-1501-v15.09.15
├── MSMT17_V1
└── VeRi
````
## Training
## Testing
## Acknowledgement
## Citation
If you find this code useful for your research, please consider citing our paper<br>
````
@inproceedings {du2023GLMC,
    title={Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions},
    author={Fei Du, peng Yang, Qi Jia, xiaoting chen, Fengtao Nan, Yun Yang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
  }
````
