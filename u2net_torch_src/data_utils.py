#### @Chao Huang(huangchao09@zju.edu.cn).

import os
import json
import math
from multiprocessing import Process, Queue
import time

import numpy as np
import pprint
from glob2 import glob
from tqdm import tqdm
import skimage
from termcolor import colored

import batchgenerators

import ccToolkits.logger as logger
from ccToolkits import logger
from ccToolkits.cc_augment import cc_augment

import config
import utils
from utils import *
import tinies

def load_files(ids, imagesDir=None, labelsDir=None):
    """
    Args:
        imagesDir: dir of raw images
        labelsDir: dir of raw labels
    Returns:
        ret: list of dicts() of im/id/gt for each id.
    """
    ret = []
    for ID in ids:
        data = {}
        data['id'] = ID
        if imagesDir is not None:
            data['im'] = os.path.join(imagesDir, ID+'.nii.gz')
        if labelsDir is not None:
            data['gt'] = os.path.join(labelsDir, ID+'.nii.gz')

        ret.append(data)
    return ret


def gen_center_bboxes(label, patch_size, config_task, margin=None):
    """
    Arg: 
        label: [d, h, w]
    Returns:
        center_bboxes, a dict of below:
        any_center_bbox: bbox of candidate centers to extract any random patches
        fore_center_bbox: bbox of candidate centers to extract patches with at least one forground class
        small_center_bbox: bbox of candidate centers to extract patches with at least one smallest class
    """
    margin = patch_size
    fore_center_bbox = []
    small_center_bbox = []

    fore_center_bbox = get_none_zero_region(np.asarray(label>0, dtype=label.dtype), margin=patch_size) # [bbmins, bbmaxes], 0,0,0,34,50,34

    # higher prob for small objs: e.g. tumor/cancer/PZ
    if config_task.task == 'Task05_Prostate':
        # PZ is smallest
        small_center_bbox = get_none_zero_region(np.asarray(label==1, dtype=label.dtype), margin=patch_size) # Task05_Prostate could have no 1(PZ) in some cases' label.
    else:
        small_center_bbox = get_none_zero_region(np.asarray(label==config_task.num_class-1, dtype=label.dtype), margin=patch_size)  # cautions! some cases have no cancer/tumor.

    center_bboxes = dict()
    if fore_center_bbox:
        center_bboxes['fore'] = [sublist[i] for i in range(len(fore_center_bbox[0])) for sublist in fore_center_bbox]
    # also works for empty list. change 0,0,0,34,50,34 to [0, 34, 0, 50, 0, 34]
    else:
        center_bboxes['fore'] = []

    if small_center_bbox:
        center_bboxes['small'] = [sublist[i] for i in range(len(small_center_bbox[0])) for sublist in small_center_bbox]
    else:
        center_bboxes['small'] = []

    return center_bboxes

def augment_patch(config_task, data, patch_size, ptype='any'):
    '''
    data augmentation and patch generation
    Args:
        data: dict of 'image', mod, d,h,w; 'label', d,h,w; 'weight', d,h,w
        patch_size: target patch_size for the model input
    Returns:
        out-of-box patch image: mod, d,h,w
    '''
    # cautions. discriminate the 2 patch sizes
    model_patch_size = config.patch_size # universal size for universal model training, otherwise, same as task specific size
    patch_size = config_task.patch_size

    image = data['image'] # mod,d,h,w
    label = data['label'] # d,h,w
    weight = data['weight'] # d,h,w
    ID = data['ID']

    # prep data to cc_augment
    image = image[np.newaxis,...]
    label = label[np.newaxis, np.newaxis,...]
    weight = weight[np.newaxis, np.newaxis,...]

    assert all([i == j for i, j in zip(image.shape[2:], label.shape[2:])]), "data and seg must have the same spatial dimensions. Data: {}, seg: {}".format(str(image.shape), str(label.shape))

    patch_img_nc, patch_lab_nc, augs = cc_augment(config_task, image, label, ptype, patch_size, patch_center_dist_from_border=[int(i/2) for i in patch_size], alpha=(0., 750.), sigma=(10., 13.), angle_x=(0, 2 * np.pi), angle_y=(0, 0), angle_z=(0, 0), scale=(0.75, 1.25), p_el_per_sample=config_task.deform_prob, p_scale_per_sample=config_task.scale_prob, p_rot_per_sample=config_task.rotate_prob, tag=config_task.task+'_'+ID)  ## patch_img_nc: 1,c,d,h,w; patch_lab_nc:1,1,d,h,w # x, y,z here are:d,h,w. # refer: ORippler/MSD_2018: 0 for angle_y and angle_z. angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi). # 
    if isinstance(augs, list):
        augs.insert(0, ID)

    if patch_img_nc is None or patch_lab_nc is None:
        patch = None
    else:
        if np.random.uniform() < config_task.mirror_prob:
            patch_img_nc[0,...], patch_lab_nc[0,...] = batchgenerators.augmentations.spatial_transformations.augment_mirroring(patch_img_nc[0,...], patch_lab_nc[0,...], axes=(2,)) # 0,1,2:d,h,w . # sample_data and sample_seg to augment_mirroring should be either [channels, x, y] or [channels, x, y, z]
            augs.append('mirror')

        patch_img = patch_img_nc[0,...]
        patch_lab = patch_lab_nc[0,0,...]
        patch_weight = None # TBD. to be included in cc_augment()
        #-----------------------------------------------------------
        ## make sure the patch size is the one for the model
        #-----------------------------------------------------------
        if config.unifyPatch == 'resize':
            # resize all tasks patches to same size for shared/universal model
            if config.trainMode in ["shared", "universal"]:
                img_list = []
                for i in range(patch_img.shape[0]):
                    img_list.append(skimage.transform.resize(patch_img[i], output_shape=model_patch_size, order=3, mode='constant')) # bi-cubic. 
                patch_img = np.asarray(img_list)
                if patch_weight is None:
                    pass
                else:
                    patch_weight = skimage.transform.resize(patch_weight, output_shape=model_patch_size, order=0, mode='constant') #0/1, so use nearest-neighbor
                patch_lab = skimage.transform.resize(patch_lab, output_shape=model_patch_size, order=0, mode='constant') # nearest-neighbor
        else:
            raise ValueError('{}: not yet implemented!!'.format(config.unifyPatch))

        patch = dict()
        patch['image'] = patch_img # mod,d,h,w
        patch['label'] = patch_lab # d, h, w
        patch['weight'] = patch_weight # d,h,w
        patch['augs'] = augs # augment methods applied for these patches
    
    return patch

def trQueue(config_task, ids, dataQueue, patch_size, nProc=1, seed=1):
    # data, center_bboxes
    '''
    args:
        file_generator to get data: dict of: 'image', mod, d,h,w; 'label',d,h,w;'weight',d,h,w.
        center_bboxes: dict of 'small','fore', 'any'.
        # batch_num: num of batches to extract
    returns:
        queue of batches.
    '''
    patch_size = config_task.patch_size
    files = load_files(ids)
    # final_files = files[1:20] # debug
    max_repeats = math.ceil(config.step_per_epoch*config.max_epoch/(len(files)*config_task.num_patch_per_file))
    # max_repeats = 10
    final_files = []

    # np.random.seed(1)
    np.random.seed(seed)
    for i in range(max_repeats):
        np.random.shuffle(files)
        final_files.extend(files)

    datDir = os.path.join(config.prepData_dir, config_task.task, "Tr")

    for obj in final_files:
        ID = obj['id']
        st_time = time.time()
        try:
            t_wait = 0
            while dataQueue.qsize() == config_task.queue_size:
                time.sleep(1)
                t_wait += 1
            if t_wait > 0:
                logger.info('{} queue is full, size={}, time waited for full:{}'.format(config_task.task, config_task.queue_size, t_wait))

            # ID = 'prostate_16' # debugging.
            # tinies.ForkedPdb().set_trace()
            
            volumes = np.load(os.path.join(datDir, ID+'_volumes.npy')) #mod, d, h, w # the largest liver case "liver_22_volumes.npy" costs 0.6s
            # volume_list = [volumes[i] for i in range(volumes.shape[0])]
            label = np.load(os.path.join(datDir, ID+'_label.npy')) # also works for NoneType obj.
            weight = np.load(os.path.join(datDir, ID+'_weight.npy'))
            
            # logger.info('ID:{}; load .npy time elapsed:{}'.format(ID, tinies.timer(st_time, time.time())))
            st_time = time.time()

            v_shape = volumes.shape
            l_shape = label.shape
            # for tasks like Task04_Hippocampus, some images smaller than patch_size, padding to patch_size, during eval, after CNN output, use crop to recover to original size.

            volume_list = []
            for moda in range(volumes.shape[0]):
                sub_vol, pad_size = tinies.pad2gePatch(volumes[moda], config_task.patch_size, data_channel=None)
                volume_list.append(sub_vol)
            volumes = np.asarray(volume_list)
            label, pad_size = tinies.pad2gePatch(label, config_task.patch_size, data_channel=None) #  could be changed to pad symmetrically instead of asymmetrically. TBD.
            weight, pad_size = tinies.pad2gePatch(weight, config_task.patch_size, data_channel=None)

            assert all([i == j for i, j in zip(volumes.shape[1:], label.shape[0:])]), "ID:{}, before pad, volumes shape:{}, label shape:{}; after pad: volumes shape:{}, label shape:{}".format(ID, str(v_shape), str(l_shape), str(volumes.shape), str(label.shape))

            data = dict()
            data['ID'] = ID
            data['image'] = volumes
            data['label'] = label
            data['weight'] = weight
            
            for i in range(config_task.num_patch_per_file):
                patches = dict()
                patches['any'] = augment_patch(config_task, data, config.patch_size,ptype='any')
                patches['fore'] = augment_patch(config_task, data, config.patch_size, ptype='fore')
                patches['small'] = augment_patch(config_task, data, config.patch_size, ptype='small')
                # patches['small'] = None
                dataQueue.put(patches)
        except Exception as e:
            logger.info('error in for-loop of trQueue:{}'.format(str(e)))

        # logger.info('ID:{}; augment_patch time elapsed:{}'.format(ID, tinies.timer(st_time, time.time())))


def sampler3d_whole(volume_list, label, weight, original_shape, bbox, im_path, gt):
    """
    volume_list: [(d,h,w)*modality]
    """
    sub_data = np.asarray(volume_list)
    out = {}
    axis = [1,2,3,0] #[1,2,3,0] [d, h, w, modalities]
    out['image']  = np.transpose(sub_data, axis)
    out['weight'] = np.transpose(weight[np.newaxis, ...], axis)
    out['original_shape'] = original_shape
    out['bbox'] = bbox
    out['im_path'] = im_path
    if gt is not None:
        out['gt_path'] = gt
    else:
        out['gt_path'] = None
    
    sub_data = None
    
    return out


def get_eval_data(obj, datDir):
    '''
    apply to one patient.
    '''
    ID = obj['id']
    gt_path = obj['gt']
    img_path = obj['im']

    volumes = np.load(os.path.join(datDir, ID+'_volumes.npy'))
    volume_list = [volumes[i] for i in range(volumes.shape[0])]
    

    label = np.load(os.path.join(datDir, ID+'_label.npy')) # also works for NoneType obj.
    weight = np.load(os.path.join(datDir, ID+'_weight.npy'))

    with open(os.path.join(datDir, ID+'.json')) as f:
        json_info = json.load(f)
    original_shape = eval(json_info['original_shape']) # eval() to unstr str type
    bbox = eval(json_info['bbox']) # eval() to unstr str type

    out = sampler3d_whole(volume_list, label, weight, original_shape, bbox, img_path, gt_path)
    
    for key in out.keys():
        if isinstance(out[key], np.ndarray):
            out[key] = np.ascontiguousarray(out[key])
    

    return out

