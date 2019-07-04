# 3D U<sup>2</sup>-Net: A 3D Universal U-Net for Multi-Domain Medical Image Segmentation

by Chao Huang, Qingsong Yao, Hu Han, Shankuan Zhu, Shaohua Zhou. This is a repo of code of the paper early accepted by MICCAI2019. The paper will be available online soon and the full code will be public then.
### In case of any questions about this repo, please feel free to contact Chao Huang(huangchao09@zju.edu.cn).

**Abstract**. Fully convolutional neural networks like U-Net have been the state-of-art methods in medical image segmentation. Practically, a network is highly specialized and trained separately for each segmenta- tion task. Instead of a collection of multiple models, it is highly desirable to learn a universal data representation for different tasks, ideally a sin- gle model with the addition of a minimal number of parameters to steer to each task. Inspired by the recent success of multi-domain learning in image classification, for the first time we explore a promising universal architecture that can handle multiple medical segmentation tasks, re- gardless of different organs and imaging modalities. Our 3D Universal U-Net (3D U2-Net) is built upon separable convolution, assuming that images from different domains have domain-specific spatial correlations which can be probed with channel-wise convolution while also share cross- channel correlations which can be modeled with pointwise convolution. We evaluate the 3D U2-Net on five organ segmentation datasets. Experimen- tal results show that this universal network is capable of competing with traditional models in terms of segmentation accuracy, while requiring only 1% of the parameters. Additionally, we observe that the architecture can be easily and effectively adapted to a new domain without sacrificing performance in the domains used to learn the shared parameterization of the universal network.

## Overview
This repo consists the codes used to train the models in our paper. Requirements are listed in `requirments.txt`. `train_model_no_adapters.py` is the mainfile to train the independent models as well as the shared model. `train_model_wt_adapters.py` is the mainfile to train the universal model with depthwise convolusion.
Terminal commands to train all models are presented in `train_models.sh`.

## Citation
If you use our code, please cite our paper:

    @inproceedings{**TBD**
    }

## Acknowledgement
We give a lot of thanks to the open-access data science community for the public data science knowledge. Special thanks are given to @Guotai Wang and @Rebuffi as some of our code are borrowed from their repos: https://github.com/taigw/brats17/ and https://github.com/srebuffi/residual_adapters.