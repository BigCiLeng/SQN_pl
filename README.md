<!--
 * @Author: BigCiLeng && bigcileng@outlook.com
 * @Date: 2023-07-25 11:49:49
 * @LastEditors: BigCiLeng && bigcileng@outlook.com
 * @LastEditTime: 2023-07-26 13:10:41
 * @FilePath: /RandLA-Net-Pytorch-New/README.md
 * @Description: 
 * 
 * Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
-->
# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

This repository contains a PyTorch-lightning implementation of [RandLA-Net](http://arxiv.org/abs/1911.11236) on S3DIS and Semantickitti.

**This repository is mainly based on the [repository](https://github.com/mellody11/RandLA-Net-Pytorch-New)**

## Preparation(S3DIS for example)

1. Clone this repository
2. Install some Python dependencies, such as scikit-learn. All packages can be installed with pip.
3. env : ubuntu 18.04, python 3.7.16, torch 1.12.1, numpy 1.21.5, torchvision 0.13.1, scikit-learn 0.22.2, pandas 1.3.5, tqdm 4.64.1
4. Install python functions. the functions and the codes are copied from the [official implementation with Tensorflow](https://github.com/QingyongHu/RandLA-Net).

```
sh compile_op.sh
```

5. Attention: please check out *./utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.7-linux-x86_64.egg/* and copy the **.so** file to the parent folder **(update in 2023.2.23: We provide a **.so** file for python3.7, and you don't need to copy(even compile the cpp code) if you are using python3.7)**
6. Download the Stanford3dDataset_v1.2_Aligned_Version[ dataset](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1), and preprocess the data:

```
  python utils/data_prepare_s3dis.py
```

   Note: Please change the dataset path in the 'data_prepare_s3dis.py' with your own path.

## Train a model(S3DIS for example)

```
cd scripts/
sh train_S3DIS.sh
```

## Test a model(S3DIS for example)

```

```

## Results

### S3DIS


### SemanticKITTI

