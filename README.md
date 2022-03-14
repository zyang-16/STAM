# STAM
Source code and dataset for WWW'22 paper "STAM: A Spatiotemporal Aggregation Method for Graph Neural Network-based Recommendation"


STAM: A Spatiotemporal Aggregation Method for Graph Neural Network-based Recommendation

Zhen Yang, Ming Ding, Bin Xu, Hongxia Yang and Jie Tang

In WWW 2022 


## Introduction
STAM is a novel aggregation method, which generates spatiotemporal neighbor embeddings from the perspectives of spatial structure information and temporal information, facilitating the development of aggregation methods from spatial to spatiotemporal. STAM utilizes the Scaled Dot-Product Attention to capture temporal orders of one-hop neighbors and employs multi-head attention to perform joint attention over different latent subspaces.

## Preparation
* Python 3.7
* Tensorflow 1.14.0


## Training
### Training on the existing datasets
You can use ```$ ./experiments/***.sh``` to train STAM model. For example, if you want to train on the ml-1m dataset, you can run ```$./experiments/train_ml.sh``` to train STAM model.




## Cite

Please cite our paper if you find this code useful for your research:
```
@article{yang2022stam,
  title={STAM: A Spatiotemporal Aggregation Method for Graph Neural Network-based Recommendation},
  author={Yang, Zhen and Ding, Ming and Bin, Xu and Yang, Hongxia and Tang, Jie},
  year={2022}
}
```
