# [3D U<sup>2</sup>-Net: A 3D Universal U-Net for Multi-Domain Medical Image Segmentation](https://link.springer.com/chapter/10.1007%2F978-3-030-32245-8_33)

by Chao Huang, Qingsong Yao, Hu Han, Shankuan Zhu, Shaohua Zhou. This is a code repo of the paper early accepted by MICCAI2019.
#### In case of any questions about this repo, please feel free to contact Chao Huang(huangchao09@zju.edu.cn).

**Abstract**. Fully convolutional neural networks like U-Net have been the state-of-art methods in medical image segmentation. Practically, a network is highly specialized and trained separately for each segmenta- tion task. Instead of a collection of multiple models, it is highly desirable to learn a universal data representation for different tasks, ideally a sin- gle model with the addition of a minimal number of parameters to steer to each task. Inspired by the recent success of multi-domain learning in image classification, for the first time we explore a promising universal architecture that can handle multiple medical segmentation tasks, re- gardless of different organs and imaging modalities. Our 3D Universal U-Net (3D U2-Net) is built upon separable convolution, assuming that images from different domains have domain-specific spatial correlations which can be probed with channel-wise convolution while also share cross- channel correlations which can be modeled with pointwise convolution. We evaluate the 3D U2-Net on five organ segmentation datasets. Experimen- tal results show that this universal network is capable of competing with traditional models in terms of segmentation accuracy, while requiring only 1% of the parameters. Additionally, we observe that the architecture can be easily and effectively adapted to a new domain without sacrificing performance in the domains used to learn the shared parameterization of the universal network.


## Overview
Brief instruction to apply the code: 
1. Most requirements are listed in `requirments.txt`. Besides, `batchgenerators` is a python package developed by researchers at the Division of Medical Image Computing at the German Cancer Research Center (DKFZ) to do data augmentation for images. Introduction and installation instructions are listed in[MIC-DKFZ/batchgenerators/](https://github.com/MIC-DKFZ/batchgenerators/).
2. Please put the datasets downloaded from [Medical Segmentation Decathlon](http://medicaldecathlon.com/) in `dataset`.
3. `data_explore.py` is to explore the characteristics of the images, e.g. pixel spacings.
4. `preprocess_taskSep.py` is used to do offline preprocessing (e.g. cropping, resampling) of the data samples to save time for training.
5. `train_model_no_adapters.py` is the mainfile to train the independent models as well as the shared model. 
6. `train_model_wt_adapters.py` is the mainfile to train the propsed universal model with separable convolution.
7. Terminal commands to train all models are presented in `train_models.sh`.

To accelerate training, we built a fast tool to do online image augmentation with CUDA on GPU(especially for elastic deformation). [**cuda_spatial_defrom**](https://github.com/qsyao/cuda_spatial_deform).

## Citation
If you use this code, please cite our paper as:

    Huang C., Han H., Yao Q., Zhu S., Zhou S.K. (2019) 3D U 2-Net: A 3D Universal U-Net for Multi-domain Medical Image Segmentation. In: Shen D. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2019. MICCAI 2019. Lecture Notes in Computer Science, vol 11765. Springer, Cham

## Acknowledgement
We give a lot of thanks to the open-access data science community for the public data science knowledge. Special thanks are given to @Guotai Wang and @Rebuffi as some of the code is borrowed from their repos: https://github.com/taigw/brats17/ and https://github.com/srebuffi/residual_adapters.