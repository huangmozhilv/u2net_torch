#### @Chao Huang(huangchao09@zju.edu.cn).
import random
import os
import copy

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import exposure, measure, morphology

import config
from ccToolkits import logger
import tinies

def preprocess(im, gt, config_task, with_gt=True):
    sitk_image = sitk.ReadImage(im)
    orig_volume = sitk.GetArrayFromImage(sitk_image) # mod, z, y, x
    if sitk_image.GetDimension() == 3:
        mod_num = 1
    elif sitk_image.GetDimension() == 4:
        mod_num = sitk_image.GetSize()[3]

    if mod_num == 1:
        orig_volume = orig_volume[np.newaxis,...]

    volume_list = []
    for mod_idx in range(mod_num):
        volume = orig_volume[mod_idx,...]
        original_shape = volume.shape
        # 155 244 244
        if mod_idx == 0:
            # contain whole tumor
            margin = 5 # small padding value
            bbmin, bbmax = get_none_zero_region(volume, margin) 
        volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
        volume = tinies.resample2fixedSpacing(volume, config_task.pixel_spacing, im, interpolate_method=sitk.sitkBSpline) # sitk.sitkLinear # cautions! remember to inversely resample the label map to cropped volume size.
        # intensity clipping
        volume[volume<-1024] = -1024 # works for CT

        if mod_idx == 0:
            weight = np.asarray(volume > 0, np.float32)

        p_l,p_u = np.percentile(volume, (2.0, 98.0))
        volume = np.clip(volume, p_l,p_u)
        if config.intensity_norm == 'modality':
            volume = itensity_normalize_one_volume(volume)
        
        volume_list.append(volume)
    

    if with_gt:
        label = sitk.GetArrayFromImage(sitk.ReadImage(gt)) # mod, d, h, w
        label[label > config_task.num_class-1] = 0 # Task04_Hippocampus 003 and 243 have one wrong gt pixel assigned 254. Here arbitrarily set to 0 （background).
        label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax) 
        # resampling.
        label = tinies.resample2fixedSpacing(label, config_task.pixel_spacing, im, interpolate_method=sitk.sitkNearestNeighbor) # cautions! remember to inversely resample the label map to cropped volume size. as indicated in last code line. # also use im as refer path, so there will no any rounding issues with inconsistence of the pixspacings

        return volume_list, label, weight, original_shape, [bbmin, bbmax]
    else:
        return volume_list, None, weight, original_shape, [bbmin, bbmax]

def get_largest_two_component(img, threshold = None, tag = ''):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D volume, prediction
        threshold: a size threshold
    outputs:
        out_img: the output volume 
    """
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling. each feature (a group of connected pixels as defined by structure) is labeled with a unique integer.
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) # cal num of pixels as labeld in labeled_array, i.e. num of pixels for each connected component.
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if len(sizes) == 0:
        logger.warn('tag:{}, component sizes:{}, np.unique(img, return_counts=True):{}'.format(tag, str(sizes_list), str(np.unique(img, return_counts=True)))) # all are background?
        out_img = img
    elif len(sizes) == 1:
        out_img = img # got only one connected component?
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0
            out_img = component1

    return out_img

def get_largest_one_component(img, threshold = None, tag = ''):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D volume, prediction
        threshold: a size threshold
    outputs:
        out_img: the output volume 
    """
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling. each feature (a group of connected pixels as defined by structure) is labeled with a unique integer.
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1)) # cal num of pixels as labeld innn labeled_array, i.e. num of pixels for each connected component.
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if len(sizes) == 0:
        logger.warn('component sizes:{}, np.unique(img, return_counts=True):{}'.format(str(sizes_list), str(np.unique(img, return_counts=True)))) # all are background?
        out_img = img
    elif len(sizes) == 1:
        out_img = img # got only one connected component?
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            component1 = labeled_array == max_label1
            out_img = component1
    
    return out_img


def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max


def set_ND_volume_roi_with_bounding_box_range(config_task, volume, bb_min, bb_max, sub_volume, interpolate_method, refer_file_path):
    """
    set a subregion to an nd image.
    bb_min: z,y,x
    bb_max: z,y,x
    """
    # cautions! remember to inversely resample the label map to original scale as vnet.
    dim = len(bb_min)
    out = volume
    

    bb_min = np.asarray(bb_min, dtype=int)
    bb_max = np.asarray(bb_max, dtype=int)
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = tinies.resample2fixedSize(sub_volume, config_task.pixel_spacing, [bb_max[0] + 1-bb_min[0], bb_max[1] + 1-bb_min[1], bb_max[2] + 1-bb_min[2]], refer_file_path, interpolate_method=interpolate_method) # when call , specify the interpolation method for label or prob
    elif(dim == 3):
        # volume: z,y,x. called for 'final_label'.
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = tinies.resample2fixedSize(sub_volume, config_task.pixel_spacing, [bb_max[0] + 1-bb_min[0], bb_max[1] + 1-bb_min[1], bb_max[2] + 1-bb_min[2]], refer_file_path, interpolate_method=interpolate_method) # apply sitkNearestNeighbor
    elif(dim == 4):
        # volume: z,y,x,num_class. called for 'final_probs'.
        for mod in range(bb_min[3], bb_max[3] + 1):
            out[bb_min[0]:(bb_max[0] + 1), bb_min[1]:(bb_max[1] + 1), bb_min[2]:(bb_max[2] + 1), mod] = tinies.resample2fixedSize(sub_volume[...,mod], config_task.pixel_spacing, [bb_max[0] + 1-bb_min[0], bb_max[1] + 1-bb_min[1], bb_max[2] + 1-bb_min[2]], refer_file_path, interpolate_method=interpolate_method) # apply sitkLinear
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out


def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape   
    patch_shape = sub_volume.shape
    output_volume = volume
    
    
    for i in range(len(center)):
        if(center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] += \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] += \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")        
    return output_volume  

def binary_dice3d(s,g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice

def get_none_zero_region(im, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = im.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(im)
    if len(indxes[0]):
        idx_min = []
        idx_max = []
        # logger.info('indxes:{}'.format(indxes))
        for i in range(len(input_shape)):
            idx_min.append(indxes[i].min())
            idx_max.append(indxes[i].max())

        for i in range(len(input_shape)):
            idx_min[i] = max(idx_min[i] - margin[i], 0)
            idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
        return idx_min, idx_max
    else:
        # some tasks, e.g. Task03_Liver, some cases have no tumor/cancer, so no small_center_bbox to cal
        return
        

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/(std + 1e-20)
    # random normal too slow
    #out_random = np.random.normal(0, 1, size = volume.shape)
    out_random = np.zeros(volume.shape)
    out[volume == 0] = out_random[volume == 0]

    
    return out

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def get_random_roi_sampling_center(input_shape, output_shape, sample_mode='full', bounding_box = None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if(sample_mode[i] == 'full'):
            if(bounding_box):
                x0 = bounding_box[i*2]; x1 = bounding_box[i*2 + 1]
            else:
                x0 = 0; x1 = input_shape[i]
        else:
            # valid
            if(bounding_box):
                x0 = bounding_box[i*2] + int(output_shape[i]/2)   
                x1 = bounding_box[i*2+1] - int(output_shape[i]/2)
            else:
                x0 = int(output_shape[i]/2)   
                x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center

def extract_roi_from_volume(volume, in_center, output_shape, fill = 'zero'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def save_to_nii(im, filename, refer_file_path, outdir="", mode="image", system="sitk", prefix=''):
    """
    Goal---save predicted mask to nii.gz with the same header of gt file.
    refer_file_path: reference file path.
    Save numpy array to nii.gz format to submit
    im: 3d numpy array ex: [908, 512, 512]
    """
    sitk_refer = sitk.ReadImage(refer_file_path)
    # extract first modality as sitk_refer if there are multiple modalities
    if sitk_refer.GetDimension() == 4:
        sitk_refer = sitk.Extract(sitk_refer, (sitk_refer.GetSize()[0], sitk_refer.GetSize()[1], sitk_refer.GetSize()[2], 0), (0,0,0,0))
    if system == "sitk":
        if mode == 'label':
            img = sitk.GetImageFromArray(im.astype(np.uint8))
        else:
            img = sitk.GetImageFromArray(im.astype(np.float32))
        
        
        img.CopyInformation(sitk_refer) # Copies the Origin, Spacing, and direction from the source image to this image, on condition that the two have the same size.
        

        writing = sitk.WriteImage(img, "./{}/{}".format(outdir, prefix + filename))
        
        
    elif system == 'nib':
        # nib_refer = nib.load(refer_file_path)
        # new_img = nib.Nifti1Image(im, nib_refer.affine, nib_refer.header)
        # nib.save(new_img, "./{}/{}".format(outdir, prefix + filename))
        
        pass
