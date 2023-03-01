# GLMC
[CVPR2023] Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions

## Overview
![image](https://user-images.githubusercontent.com/48430480/222028170-e63da465-e143-4c6d-bdb9-ca1b3e31d469.png)


> We propose an efficient one-stage training strategy for long-tailed visual recognition called Global and Local Mixture Consistency cumulative learning (GLMC). Our core ideas are twofold: (1) a global and local mixture consistency loss improves the robustness of the feature extractor. Specifically, we generate two augmented batches by the global MixUp and local CutMix from the same batch data, respectively, and then use cosine similarity to minimize the difference. (2)A cumulative head-tail soft label reweighted loss mitigates the head class bias problem. We use empirical class frequencies to reweight the mixed label of the head-tail class for long-tailed data and then balance the conventional loss and the rebalanced loss with a coefficient accumulated by epochs.

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
