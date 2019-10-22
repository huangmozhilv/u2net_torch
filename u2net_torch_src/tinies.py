#### @Chao Huang(huangchao09@zju.edu.cn).
# a custom package of small utilities compiled by Chao.

import os
import sys
import pdb
import itertools
import time
import shutil
import json

import numpy as np

from glob2 import glob
import SimpleITK as sitk

from ccToolkits import logger
import config

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    e.g. ForkedPdb().set_trace()
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def newdir(path):
    # always make new dir
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def sureDir(path):
    # only make new dir when not existing
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def timer(start, end):
    '''
    end-start: returns seconds
    '''
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "hr:min:sec, {:0}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds))


def calPatchWeights(patch_size, distType='Eu'):
    '''
    Args:
        patch_size: list
    Return:
        patch_weights: numpy array of shape of patch_size
    e.g:
    p128 = tinies.calPatchWeights([128, 128, 128])
    p128[0,0,0] # 0
    p128[0,0,1] # 0.005
    p128[127,0,0] # 0.005
    p128[100,100,100] # 0.4375
    p128[64,64,64] # 1
    '''
    # weights defined as inversely related to euclidean distance btw one point and the center. Closer to center, weigh more.
    patch_weights = dists = np.zeros(patch_size)

    def euDis(s):
        dis = np.sqrt(np.sum(np.asarray([np.square(float(s[0])-patch_size[0]/2), np.square(float(s[1])-patch_size[1]/2), np.square(float((s[2]))-patch_size[2]/2)])))
        return dis

    # start = time.time()
    dists_list = list(map(euDis, itertools.product(range(patch_size[0]), range(patch_size[1]), range(patch_size[2]))))
    dists = np.asarray(dists_list).reshape(patch_size)
    patch_weights  = 1-(dists-dists.min())/(dists.max()-dists.min()) + 1e-20 # norm to (0,1)
    # print("patch_weights cal time:{}".format(time.time()-start)) # 29.19s
    # dists = (dists-dists.min())/(dists.max()-dists.min())
    # patch_weights = np.sqrt(1-np.power(dists, 2) +0.000001)
    return patch_weights


def pad2gePatch(img, patch_size, data_channel=None):
    '''
    for tasks like Task04_Hippocampus, some cases have images smaller than patch_size. In this case, use this function to pad to patch_size or larger, during eval, after CNN output, use crop to recover to original size.
    '''
    patch_size = np.asarray(patch_size)
    if data_channel:
        subimg_shape = np.asarray(img[0].shape)
    else:
        subimg_shape = np.asarray(img.shape)
    pad_size = patch_size-subimg_shape
    pad_size = np.clip(pad_size, 0, None)
    half_pad_size = [int(i) for i in np.ceil(pad_size/2)]
    padded_size = subimg_shape + np.asarray(half_pad_size)*2

    new_subimg_shape = np.max([subimg_shape, padded_size], axis=0)
    if np.any(pad_size):
        if data_channel:
            new_img = np.zeros([data_channel]+list(new_subimg_shape))
            for ch in range(data_channel):
                # new_img[ch] = np.pad(img[ch], ((0, pad_size[0]), (0, pad_size[1]), (0, pad_size[2])), mode='constant', constant_values = 0) # img[ch].min()-1000
                new_img[ch] = np.pad(img[ch], ((half_pad_size[0], half_pad_size[0]), (half_pad_size[1], half_pad_size[1]), (half_pad_size[2], half_pad_size[2])), mode='constant', constant_values = 0) # img[ch].min()-1000
        else:
            new_img = np.zeros(list(new_subimg_shape))
            # new_img = np.pad(img, ((0, pad_size[0]), (0, pad_size[1]), (0, pad_size[2])), mode='constant', constant_values = 0) # img.min()-1000
            new_img = np.pad(img, ((half_pad_size[0], half_pad_size[0]), (half_pad_size[1], half_pad_size[1]), (half_pad_size[2], half_pad_size[2])), mode='constant', constant_values = 0) # img.min()-1000
    else:
        new_img = img
    # return new_img, pad_size
    return new_img, half_pad_size


def resample2fixedSpacing(volume, newSpacing, refer_file_path, interpolate_method=sitk.sitkBSpline): 
    # sitk.sitkLinear
    '''
    also works for 2-D?
    Resample dat(i.e. one 3-D sitk image/GT) to destine resolution. but keep the origin, direction be the same.
    Volume: 3D numpy array, z, y, x.
    oldSpacing: z,y,x
    newSpacing: z,y,x
    refer_file_path: source to get origin, direction, and oldSpacing. Here we use the image_file path.
    ''' 
    # in the project, oldSpacing, origin, direction will be extracted from gt_file as the refer_file_path
    sitk_refer = sitk.ReadImage(refer_file_path)
    # extract first modality as sitk_refer if there are multiple modalities
    if sitk_refer.GetDimension() == 4:
        sitk_refer = sitk.Extract(sitk_refer, (sitk_refer.GetSize()[0], sitk_refer.GetSize()[1], sitk_refer.GetSize()[2], 0), (0,0,0,0))
    origin = sitk_refer.GetOrigin()
    oldSpacing =  sitk_refer.GetSpacing()
    direction = sitk_refer.GetDirection()
    

    # prepare oldSize, oldSpacing, newSpacing, newSize in order of [x,y,z]
    oldSize = np.asarray(volume.shape, dtype=float)[::-1]
    oldSpacing = np.asarray([round(i, 3) for i in oldSpacing], dtype=float)
    newSpacing = np.asarray([round(i, 3) for i in newSpacing], dtype=float)[::-1]
    # compute new size, assuming same volume of tissue (not number of total pixels) before and after resampled 
    newSize = np.asarray(oldSize * oldSpacing/newSpacing, dtype=int)

    # create sitk_old from array and set appropriate meta-data
    sitk_old = sitk.GetImageFromArray(volume)
    

    sitk_old.SetOrigin(origin)
    sitk_old.SetSpacing(oldSpacing)
    sitk_old.SetDirection(direction)
    sitk_new = sitk.Resample(sitk_old, newSize.tolist(), sitk.Transform(), interpolate_method, origin, newSpacing, direction)

    newVolume = sitk.GetArrayFromImage(sitk_new)

    
    return newVolume


def resample2fixedSize(volume, oldSpacing, newSize, refer_file_path, interpolate_method=sitk.sitkNearestNeighbor):
    '''
    also works for 2-D?
    Goal---resample to fixed size with new spacing, but keep the origin, direction be the same.
    volume: 3-D numpy array, z, y, x. In this code package, this is the final predicted label map, its shape is that of cropped non-zero region resampled to fixed spacings. 
    newSize: z,y,x. in this project, its shape is that of the cropped non-zero region.
    oldSpacing: z,y,x. the spacing of the volume. In this project, it's the isotropical spacing.
    refer_file_path: source to get origin, direction, and newSpacing. Here we use the image_file path.
    ''' 
    # in the project, newSpacing, origin, direction will be extracted from gt_file as the refer_file_path
    sitk_refer = sitk.ReadImage(refer_file_path)
    # extract first modality as sitk_refer if there are multiple modalities
    if sitk_refer.GetDimension() == 4:
        sitk_refer = sitk.Extract(sitk_refer, (sitk_refer.GetSize()[0], sitk_refer.GetSize()[1], sitk_refer.GetSize()[2], 0), (0,0,0,0))
    origin = sitk_refer.GetOrigin()
    newSpacing =  sitk_refer.GetSpacing()
    direction = sitk_refer.GetDirection()
    

    # prepare oldSize, oldSpacing, newSpacing, newSize in order of [x,y,z]
    oldSpacing = np.asarray(oldSpacing, dtype=float)[::-1]
    newSpacing = np.asarray(newSpacing, dtype=float)
    # compute new size, assuming same volume of tissue (not number of total pixels) before and after resampled 
    newSize = np.asarray(newSize, dtype=int)[::-1]

    # create sitk_old from array and set appropriate meta-data
    sitk_old = sitk.GetImageFromArray(volume)
    

    sitk_old.SetOrigin(origin)
    sitk_old.SetSpacing(oldSpacing)
    sitk_old.SetDirection(direction)
    sitk_new = sitk.Resample(sitk_old, newSize.tolist(), sitk.Transform(), interpolate_method, origin, newSpacing, direction)

    newVolume = sitk.GetArrayFromImage(sitk_new)


    return newVolume

