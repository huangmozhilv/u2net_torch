#### @Chao Huang(huangchao09@zju.edu.cn).
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def torch_binary_dice(input, target, eps=0.00001):
    assert(input.shape==target.shape)
    input = input.float()
    target = target.float()

    N = target.size(0)

    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)

    intersection = input_flat * target_flat

    dice = (2*intersection.sum(1) + eps) / (input_flat.sum(1) + target_flat.sum(1) + eps)

    return dice.sum() / N

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        input = F.softmax(input, dim=1)
        num_class = input.shape[1]
        target = one_hot(target, num_class)
        
        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        total_dice = 0
        for i in range(num_class):
            dice = torch_binary_dice(input[:,i, ...], target[:,i, ...], eps = 0.000001)
            if weights is not None:
                dice *= weights[i]
            total_dice += dice
        mean_dice = torch.div(total_dice, num_class)
        # C = target.shape[1]
        # target = one_hot(target, C)
        
        # # if weights is None:
        # # 	weights = torch.ones(C) #uniform weights for all classes

        # total_dice = 0
        # for i in range(C):
        #     dice = torch_binary_dice(input[:,i, ...], target[:,i, ...], eps = 0.000001)
        #     if weights is not None:
        #         dice *= weights[i]
        #     total_dice += dice
        # mean_dice = torch.div(total_dice, C)

        return -mean_dice


def one_hot(inputs, n_classes, use_gpu=True):
    inputs = inputs.long().unsqueeze(1)

    target_shape = list(inputs.shape)
    target_shape[1] = n_classes

    output = torch.zeros(target_shape)
    if use_gpu:
        output = output.cuda()

    return output.scatter_(1, inputs, 1)
