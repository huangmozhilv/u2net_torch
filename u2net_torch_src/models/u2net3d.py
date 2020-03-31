#### @Chao Huang(huangchao09@zju.edu.cn).
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccToolkits import logger

import config
from models.model_utils import num_pool2stride_size

# u2net3d(3D U-squared Net): universally applicable unet3d.
# This code was partially inspired by nnU_Net and residual adapter paper(https://github.com/srebuffi/residual_adapters/).

# series_adapter and parallel_adapter are from https://github.com/srebuffi/residual_adapters/.
# separable_adapter is the proposed adapter in our U2Net.

'''
Input: (N, C_{in}, D_{in}, H_{in}, W_{in})
Output: (N, C_{out}, D_{out}, H_{out}, W_{out})
'''

def norm_act(nchan, only='both'):
    if config.instance_norm:
        norm = nn.InstanceNorm3d(nchan, affine=True)
    # act = nn.ReLU() # activation
    act = nn.LeakyReLU(negative_slope=1e-2)
    if only=='norm':
        return norm
    elif only=='act':
        return act
    else:
        return nn.Sequential(norm, act)


class conv1x1(nn.Module):
    def __init__(self, inChans, outChans=None, stride=1, padding=0):
        super(conv1x1, self).__init__()
        if config.module == 'series_adapter':
            self.op1 = nn.Sequential(
                norm_act(inChans,only='norm'),
                nn.Conv3d(inChans, inChans, kernel_size=1, stride=1)
                )
        elif config.module == 'parallel_adapter':
            self.op1 = nn.Conv3d(inChans, outChans, kernel_size=1, stride=stride, padding=padding)
        else:
            self.op1 = nn.Conv3d(inChans, inChans, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.op1(x)
        if config.module == 'series_adapter':
            out += x
        return out

class dwise(nn.Module):
    def __init__(self, inChans, kernel_size=3, stride=1, padding=1):
        super(dwise, self).__init__()
        self.conv1 = nn.Conv3d(inChans, inChans, kernel_size=kernel_size, stride=stride, padding=padding, groups=inChans)
        self.op1 = norm_act(inChans,only='both')

    def forward(self, x):
        out = self.conv1(x)
        out = self.op1(out)
        return out

class pwise(nn.Module):
    def __init__(self, inChans, outChans, kernel_size=1, stride=1, padding=0):
        super(pwise, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv1(x)
        return out

class conv_unit(nn.Module):
    '''
    variants of conv3d+norm by applying adapter or not.
    '''
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1, second=0):
        super(conv_unit, self).__init__()
        self.stride = stride

        if self.stride != 1:
            self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding) # padding != 0 for stride != 2 if doing padding=SAME.
        elif self.stride == 1:
            if config.trainMode != 'universal': # independent, shared
                self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding) # padding != 0 for stride != 2 if doing padding=SAME.
            else:
                if config.module in ['series_adapter', 'parallel_adapter']:
                    self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding) # padding != 0 for stride != 2 if doing padding=SAME.
                    if config.module == 'series_adapter':
                        self.adapOps = nn.ModuleList([conv1x1(outChans) for i in range(nb_tasks)]) # based on https://github.com/srebuffi/residual_adapters/
                    elif config.module == 'parallel_adapter':
                        self.adapOps = nn.ModuleList([conv1x1(inChans, outChans) for i in range(nb_tasks)]) 
                    else:
                        pass
                elif config.module == 'separable_adapter':
                    logger.info('using module of :{}'.format(config.module))
                    self.adapOps = nn.ModuleList([dwise(inChans) for i in range(nb_tasks)])
                    self.pwise = pwise(inChans, outChans)
                else:
                    pass                

        self.op = nn.ModuleList([norm_act(outChans, only='norm') for i in range(nb_tasks)])

    def forward(self, x):
        task_idx = config.task_idx
        # import ipdb; ipdb.set_trace()
        if self.stride != 1:
            out = self.conv(x)
            out = self.op[task_idx](out)
            return out
        elif self.stride == 1:
            if config.trainMode != 'universal': # independent, shared
                out = self.conv(x)
                out = self.op[task_idx](out)
            else:
                if config.module in ['series_adapter', 'parallel_adapter']:
                    out = self.conv(x)
                    if config.module == 'series_adapter':
                        out = self.adapOps[task_idx](out)
                    elif config.module == 'parallel_adapter':
                        share_map = out
                        para_map = self.adapOps[task_idx](x)
                        out = out + para_map
                    else:
                        pass

                    out = self.op[task_idx](out)
                    if config.module == 'parallel_adapter':
                        return out, share_map, para_map # for visualization of feature maps
                    else:
                        return out
                elif config.module == 'separable_adapter':
                    out = self.adapOps[task_idx](x)
                    para_map = out
                    out = self.pwise(out)
                    share_map = out
                    out = self.op[task_idx](out)
                    return out, share_map, para_map
                else:
                    pass


class InputTransition(nn.Module):
    '''
    task specific
    '''
    def __init__(self, inChans, base_outChans):
        super(InputTransition, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv3d(inChans, base_outChans, kernel_size=3, stride=1, padding=1),
            norm_act(base_outChans)
        )

    def forward(self, x):
        out = self.op1(x)
        return out

class DownSample(nn.Module):
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1):
        super(DownSample, self).__init__()
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act1 = norm_act(outChans, only="act")

    def forward(self, x):
        out = self.op1(x)
        out = self.act1(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1):
        super(DownBlock, self).__init__()
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act1 = norm_act(outChans, only="act")
        self.op2 = conv_unit(nb_tasks, outChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act2 = norm_act(outChans, only="act")

    def forward(self, x):
        if config.module == 'parallel_adapter' or config.module == 'separable_adapter':
            out, share_map, para_map = self.op1(x)
        else:
            out = self.op1(x)
        out = self.act1(out)
        if config.module == 'parallel_adapter' or config.module == 'separable_adapter':
            out, share_map, para_map = self.op2(out)
        else:
            out = self.op2(out)
        if config.residual: # same to ResNet
            out = self.act2(x + out)
        else:
            out = self.act2(out)

        return out


def Upsample3D(scale_factor=(2,2,2)):
    '''
    task specific
    '''
    # l = tf.keras.layers.UpSampling3D(size=up_strides, data_format=DATA_FORMAT)(l) # by tkuanlun350. # no equavalent in torch?
    # scale_factor can also be a tuple. so able to custom scale_factor for each dim.
    upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest') # ignore the warnings. Only module like upsample can be shown in my visualization. # if using ConvTranspose3d, be careful to how to pad when the down sample method used padding='SAME' strategy.
    return upsample

class UnetUpsample(nn.Module):
    def __init__(self, nb_tasks, inChans, outChans, up_stride=(2,2,2)):
        super(UnetUpsample, self).__init__()
        self.upsamples = nn.ModuleList(
            [Upsample3D(scale_factor=up_stride) for i in range(nb_tasks)]
        )
        self.op = conv_unit(nb_tasks, inChans, outChans, kernel_size=3,stride=1, padding=1)
        self.act = norm_act(outChans, only='act')

    def forward(self, x):
        task_idx = config.task_idx
        out = self.upsamples[task_idx](x)
        if config.module == 'parallel_adapter' or config.module == 'separable_adapter':
            out, share_map, para_map = self.op(out)
        else:
            out = self.op(out)
        out = self.act(out)
        if config.module == 'parallel_adapter' or config.module == 'separable_adapter':
            return out, share_map, para_map
        else:
            return out

class UpBlock(nn.Module):
    def __init__(self, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1):
        super(UpBlock, self).__init__()
        self.op1 = conv_unit(nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act1 = norm_act(outChans, only="act")
        self.op2 = conv_unit(nb_tasks, outChans, outChans, kernel_size=1, stride=1, padding=0)
        self.act2 = norm_act(outChans, only="act")

    def forward(self, x, up_x):
        if config.module == 'parallel_adapter' or config.module == 'separable_adapter':
            out, share_map, para_map = self.op1(x)
        else:
            out = self.op1(x)
        out = self.act1(out)
        if config.module == 'parallel_adapter' or config.module == 'separable_adapter':
            out, share_map, para_map = self.op2(out)
        else:
            out = self.op2(out)

        out = self.act2(out)

        return out

class DeepSupervision(nn.Module):
    '''
    task specific
    '''
    def __init__(self, inChans, num_class, up_stride=(2,2,2)):
        super(DeepSupervision, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv3d(inChans, num_class, kernel_size=1, stride=1, padding=0),
            norm_act(num_class)
        ) 
        self.op2 = Upsample3D(scale_factor=up_stride)

    def forward(self, x, deep_supervision):
        if deep_supervision is None:
            out = self.op1(x)
        else:
            out = torch.add(self.op1(x), deep_supervision)
        out = self.op2(out)
        return out

class OutputTransition(nn.Module):
    '''
    task specific
    '''
    def __init__(self, inChans, num_class):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, num_class, kernel_size=1, stride=1, padding=0)
       
    def forward(self, x, deep_supervision=None):
        out = self.conv1(x)
        if deep_supervision is None:
            return out
        else:
            out = torch.add(out, deep_supervision)
            return out


class u2net3d(nn.Module):
    def __init__(self, inChans_list=[2], base_outChans=16, num_class_list=[4]):
        '''
        Args:
        One or more tasks could be input at once. So lists of inital model settings are passed.
            inChans_list: a list of num_modality for each input task.
            base_outChans: outChans of the inputTransition, i.e. inChans of the first layer of the shared backbone of the universal model.
            depth: depth of the shared backbone.
        '''
        logger.info('------- base_outChans is {}'.format(base_outChans))
        super(u2net3d, self).__init__()
        
        nb_tasks = len(num_class_list)

        self.depth = max(config.num_pool_per_axis) + 1 # config.num_pool_per_axis firstly defined in train_xxxx.py or main.py
        stride_sizes = num_pool2stride_size(config.num_pool_per_axis)

        self.in_tr_list = nn.ModuleList(
            [InputTransition(inChans_list[j], base_outChans) for j in range(nb_tasks)]
        ) # task-specific input layers

        outChans_list = list()
        self.down_blocks = nn.ModuleList() # # register modules from regular python list.
        self.down_samps = nn.ModuleList()
        self.down_pads = list() # used to pad as padding='same' in tensorflow

        inChans = base_outChans
        for i in range(self.depth):
            outChans = base_outChans * (2**i)
            outChans_list.append(outChans)
            self.down_blocks.append(DownBlock(nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1))
        
            if i != self.depth-1:
                # stride for each axis could be 1 or 2, depending on tasks. # to apply padding='SAME' as tensorflow, cal and save pad num to manually pad in forward().
                pads = list() # 6 elements for one 3-D volume. originized for last dim backward to first dim, e.g. w,w,h,h,d,d # required for F.pad.
                # pad 1 to the right end if s=2 else pad 1 to both ends (s=1). 
                for j in stride_sizes[i][::-1]:
                    if j == 2:
                        pads.extend([0,1])
                    elif j == 1:
                        pads.extend([1,1])
                self.down_pads.append(pads) 
                self.down_samps.append(DownSample(nb_tasks, outChans, outChans*2, kernel_size=3, stride=tuple(stride_sizes[i]), padding=0))
                inChans = outChans*2
            else:
                inChans = outChans

        self.up_samps = nn.ModuleList([None] * (self.depth-1))
        self.up_blocks = nn.ModuleList([None] * (self.depth-1))
        self.dSupers = nn.ModuleList() # 1 elements if self.depth =2, or 2 elements if self.depth >= 3
        for i in range(self.depth-2, -1, -1):
            self.up_samps[i] = UnetUpsample(nb_tasks, inChans, outChans_list[i], up_stride=stride_sizes[i])

            self.up_blocks[i] = UpBlock(nb_tasks, outChans_list[i]*2, outChans_list[i], kernel_size=3,stride=1, padding=1)

            if config.deep_supervision and i < 3 and i > 0:
                self.dSupers.append(nn.ModuleList(
                    [DeepSupervision(outChans_list[i], num_class_list[j], up_stride=tuple(stride_sizes[i-1])) for j in range(nb_tasks)]
                ))

            inChans = outChans_list[i]

        self.out_tr_list = nn.ModuleList(
            [OutputTransition(inChans, num_class_list[j]) for j in range(nb_tasks)]
        )
        # logger.info('out_tr:\n{}'.format(str(self.out_tr)))
        

    def forward(self, x):
        # x: [N, C, D, H, W]
        task_idx = config.task_idx
        deep_supervision = None

        out = self.in_tr_list[task_idx](x)

        down_list = list()
        for i in range(self.depth):
            out = self.down_blocks[i](out)
            # down_list.append(out)
            if i != self.depth-1:
                down_list.append(out) # will not store the deepest, so as to save memory
                # manually padding='SAME' as tensorflow before down sampling
                out = F.pad(out,tuple(self.down_pads[i]), mode="constant", value=0)
                out = self.down_samps[i](out)
        
        idx = 0
        for i in range(self.depth-2, -1, -1):
            if config.module == 'parallel_adapter' or config.module == 'separable_adapter':
                out, share_map, para_map = self.up_samps[i](out)
            else:
                out = self.up_samps[i](out)
            up_x = out
            out = torch.cat((out, down_list[i]), dim=1)
            out = self.up_blocks[i](out, up_x)

            if config.deep_supervision and i < 3 and i > 0:
                deep_supervision = self.dSupers[idx][task_idx](out, deep_supervision)
                idx += 1
        out = self.out_tr_list[task_idx](out, deep_supervision)
        
        if config.module == 'parallel_adapter' or config.module == 'separable_adapter':
            return out, share_map, para_map
        else:
            return out