#### @Chao Huang(huangchao09@zju.edu.cn).

import os
import csv
import time
import random
from collections import namedtuple

import numpy as np
import SimpleITK as sitk
import skimage.transform
from skimage.transform import resize
from scipy import ndimage

import torch

from tqdm import tqdm
from ccToolkits import logger
import config
from utils import *
from data_utils import load_files, get_eval_data
import tinies
import train

def post_processing(config_task, pred_raw, temp_weight=None, ID=''):
    struct = ndimage.generate_binary_structure(3, 2)
    margin = 5
    wt_threshold = None
    if temp_weight is None:
        temp_weight = np.ones_like(pred_raw)
    pred_raw = pred_raw * temp_weight 
    
    out_label = np.zeros_like(pred_raw, dtype=np.uint8) # by Chao. VERY IMPORTANT TO AVOID out_label to be forced to 0/1 array
    for i in range(1, config_task.num_class):
        pred_tmp = np.zeros_like(pred_raw)
        pred_tmp[pred_raw==i] = 1
        if i == config_task.num_class-1 and any([x in str(config_task.labels[str(i)]).lower() for x in ['cancer', 'tumour']]):
            out_label[pred_raw==i] = i # don't apply get_largest_two_component to the highest level class (e.g. cancer)# some cases like liver cancer, there could be multiple tumors in one liver.
        else:
            pred_tmp = ndimage.morphology.binary_closing(pred_tmp, structure = struct) 
            try:
                if config_task.task in ['Task02_Heart', 'Task03_Liver', 'Task07_Pancreas', 'Task09_Spleen']:
                    if config_task.task in ['Task09_Spleen']:
                        pred_tmp[...,:,int(pred_raw.shape[-1]/2)::] = 0
                    pred_tmp = get_largest_one_component(pred_tmp, wt_threshold, ID+'_label'+str(i))
                else:
                    pred_tmp = get_largest_two_component(pred_tmp, wt_threshold, ID+'_label'+str(i))
            except:
                logger.info(' class:{}, np.uniques(pred_raw):{}'.format(i, str(np.unique(pred_raw, return_counts=True))))
                # import ipdb; ipdb.set_trace()
            out_label[pred_tmp==1] = i
               
        
    return out_label

def batch_segmentation(config_task, temp_imgs, model):
    # temp_imgs: mod_num, D, H, W?
    model_patch_size = config.patch_size # model patch size. if args.trainMode='independent', equal to config_task.patch_size; else, not equal.
    batch_size = config.batch_size
    num_class = config_task.num_class
    patch_weights = torch.from_numpy(config.patch_weights).float().cuda()
    
    data_channel, original_D, original_H, original_W = temp_imgs.shape # data_channel = 4
    
    # for some cases, e.g. Task04_Hippocampus. temp_imgs[0] shape is smaller than patch_size.. pad to patch_size. remember to apply the same process to get_train_dataflow()
    # import ipdb; ipdb.set_trace()
    temp_imgs, pad_size = tinies.pad2gePatch(temp_imgs, config_task.patch_size, data_channel)

    data_channel, D, H, W = temp_imgs.shape
    # temp_prob1 = np.zeros([D, H, W, num_class])

    ### before input to model, scale the image with factor of model_patch_size/task_specific_patch_size,so as to unify the patch size to the size required by the universal pipeline model.
    st_time = time.time()

    oldShape = [D,H,W]
    if config.unifyPatch == 'resize':
        # resize all tasks images to same size for shared/universal model
        if config.trainMode in ["shared", "universal"]:
            
            # tb visualization
            # colorslist=['#000000','#00FF00','#0000FF','#FF0000', '#FFFF00']
            tb_image = temp_imgs[0,...]
            slice_indices = [8 * i for i in range(int(tb_image.shape[0]/8))]
            img_fig = config.writer.tensor2figure(tb_image, slice_indices, colorslist=config.colorslist, is_label=False, fig_title='image')
            # config.writer.add_figure('figure/{}_batch_seg_temp_imgs_before_resize2modelpatch'.format(config_task.task), [img_fig], config.step)


            scale_factors = [model_patch_size[i]/config_task.patch_size[i] for i in range(len(model_patch_size))]
            newShape = [int(oldShape[i]*scale_factors[i]) for i in range(len(scale_factors))]
            imgs_list = []
            for i in range(temp_imgs.shape[0]):
                imgs_list.append(skimage.transform.resize(temp_imgs[i], output_shape=tuple(newShape), order=3, mode='constant')) # bi-cubic. 
            temp_imgs = np.asarray(imgs_list)

            # tb visualization
            # colorslist=['#000000','#00FF00','#0000FF','#FF0000', '#FFFF00']
            tb_image = temp_imgs[0,...]
            slice_indices = [8 * i for i in range(int(tb_image.shape[0]/8))]
            img_fig = config.writer.tensor2figure(tb_image, slice_indices, colorslist=config.colorslist, is_label=False, fig_title='image')
            # config.writer.add_figure('figure/{}_batch_seg_temp_imgs_after_resize2modelpatch'.format(config_task.task), [img_fig], config.step)

    else:
        raise ValueError('{}: not yet implemented!!'.format(config.unifyPatch))
    
    # logger.info('resize2modelpatch time elapsed:{}'.format(tinies.timer(st_time, time.time())))

    data_channel, D, H, W = temp_imgs.shape
    temp_prob1 = np.zeros([D, H, W, num_class])

    data_mini_batch = []
    centers = []
    
    st_time = time.time()

    for patch_center_W in range(int(model_patch_size[2]/2), W + int(model_patch_size[2]/2), int(model_patch_size[2]/2)):
        patch_center_W = min(patch_center_W, W - int(model_patch_size[2]/2))
        for patch_center_H in range(int(model_patch_size[1]/2), H + int(model_patch_size[1]/2), int(model_patch_size[1]/2)):
            patch_center_H = min(patch_center_H, H - int(model_patch_size[1]/2))
            for patch_center_D in range(int(model_patch_size[0]/2), D + int(model_patch_size[0]/2), int(model_patch_size[0]/2)):
                patch_center_D = min(patch_center_D, D - int(model_patch_size[0]/2))
                temp_input_center = [patch_center_D, patch_center_H, patch_center_W]
                # logger.info("temp_input_center:{}".format(temp_input_center))
                # ipdb.set_trace()
                centers.append(temp_input_center)

                patch = []
                for chn in range(data_channel):
                    sub_patch = extract_roi_from_volume(temp_imgs[chn], temp_input_center, model_patch_size, fill="zero")
                    patch.append(sub_patch)
                patch = np.asanyarray(patch, np.float32) #[mod,d,h,w]
                # collect to batch
                data_mini_batch.append(patch) # [14,4,d,h,w] # 4, modalities;

                if len(data_mini_batch) == batch_size:
                    data_mini_batch = np.asarray(data_mini_batch, np.float32)
                    # data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1]) # batch_size, d, h, w, modality
                    # ipdb.set_trace()
                    data_mini_batch = torch.from_numpy(data_mini_batch).float().cuda() # numpy to torch to GPU
                    if config.trainMode == "universal":
                        prob_mini_batch1, share_map, para_map = model(data_mini_batch)
                    else:
                        prob_mini_batch1 = model(data_mini_batch)
                    
                    # if config.test_flip:
                    #     prob_mini_batch1 += model(torch.flip(data_mini_batch, [4]))
                    
                    prob_mini_batch1 = prob_mini_batch1.detach()

                    # prob_mini_batch1 = np.transpose(prob_mini_batch1, [0,2,3,4,1]) # n,d,h,w,c
                    prob_mini_batch1 = prob_mini_batch1.permute([0,2,3,4,1]) # n,d,h,w,c

                    data_mini_batch = []
                    for batch_idx in range(prob_mini_batch1.shape[0]):
                        sub_prob = prob_mini_batch1[batch_idx]

                        for i in range(num_class):
                            # sub_prob[...,i] = np.multiply(sub_prob[...,i], config.patch_weights)
                            sub_prob[...,i] = torch.mul(sub_prob[...,i], patch_weights)
                        
                        sub_prob = sub_prob.cpu().numpy()

                        temp_input_center = centers[batch_idx]
                        for c in range(num_class):
                            temp_prob1[...,c] = set_roi_to_volume(temp_prob1[...,c], temp_input_center, sub_prob[...,c])
                    centers = []
    

    remainder_batch_size = len(data_mini_batch)
    if remainder_batch_size > 0 and remainder_batch_size < batch_size:
        # treat the remainder as an idependent batch as it's smaller than batch_size
        for idx in range(batch_size-len(data_mini_batch)):
            data_mini_batch.append(np.zeros([data_channel]+model_patch_size)) # fill to full batch_size with zeros array
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        # data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1]) # batch_size, d, h, w, modality

        data_mini_batch = torch.from_numpy(data_mini_batch).float().cuda() # numpy to torch to GPU
        if config.trainMode == "universal":
            prob_mini_batch1, share_map, para_map = model(data_mini_batch)
        else:
            prob_mini_batch1 = model(data_mini_batch)
        # if config.test_flip: # flip on w axis?
        #     prob_mini_batch1 += model(torch.flip(data_mini_batch, [4]))
        prob_mini_batch1 = prob_mini_batch1.detach()
        # prob_mini_batch1 = np.transpose(prob_mini_batch1, [0,2,3,4,1])
        prob_mini_batch1 = prob_mini_batch1.permute([0,2,3,4,1]) # n,d,h,w,c
        # logger.info('prob_mini_batch1 shape:{}'.format(prob_mini_batch1.shape))

        data_mini_batch = []
        for batch_idx in range(remainder_batch_size):
            sub_prob = prob_mini_batch1[batch_idx]
            # sub_prob = np.reshape(prob_mini_batch1[batch_idx], model_patch_size + [num_class])

            for i in range(num_class):
                # sub_prob[...,i] = np.multiply(sub_prob[...,i], config.patch_weights)
                sub_prob[...,i] = torch.mul(sub_prob[...,i], patch_weights)
            
            sub_prob = sub_prob.cpu().numpy()

            temp_input_center = centers[batch_idx]
            for c in range(num_class):
                temp_prob1[...,c] = set_roi_to_volume(temp_prob1[...,c], temp_input_center, sub_prob[...,c])
    elif remainder_batch_size >= batch_size:
        logger.error('the remainder data_mini_batch size is {} and batch_size = {}, code is wrong'.format(len(data_mini_batch), batch_size))
    
    logger.info('patch eval for-loop time elapsed:{}'.format(tinies.timer(st_time, time.time())))

    # argmax
    temp_pred1 = np.argmax(temp_prob1, axis=-1)
    # temp_pred1 = np.asarray(temp_pred1, dtype=np.uint8)
    
    if config.unifyPatch == 'resize':
        # resize all tasks images to same size for universal model
        if config.trainMode in ["shared", "universal"]:

            # tb visualization
            # colorslist=['#000000','#00FF00','#0000FF','#FF0000', '#FFFF00']
            tb_image = temp_imgs[0,...]
            tb_pred = temp_pred1
            slice_indices = config.writer.chooseSlices(tb_pred)
            img_fig = config.writer.tensor2figure(tb_image, slice_indices, colorslist=config.colorslist, is_label=False, fig_title='image')
            pred_fig = config.writer.tensor2figure(tb_pred, slice_indices, colorslist=config.colorslist, is_label=True, fig_title='pred')
            # config.writer.add_figure('figure/{}_batch_seg_temp_pred1_before_resize2originalScale'.format(config_task.task), [img_fig, pred_fig], config.step)

            # reisze
            temp_pred1 = temp_pred1.astype(np.float32) # it will result in nothing if input an array of np.uint8 to resize(order=0)
            temp_pred1 = skimage.transform.resize(temp_pred1, output_shape=tuple(oldShape), order=0, mode='constant') # nearest.

            # tb visualization
            # colorslist=['#000000','#00FF00','#0000FF','#FF0000', '#FFFF00']
            tb_image = temp_imgs[0,...]
            tb_pred = temp_pred1
            slice_indices = config.writer.chooseSlices(tb_pred)
            img_fig = config.writer.tensor2figure(tb_image, slice_indices, colorslist=config.colorslist, is_label=False, fig_title='image')
            pred_fig = config.writer.tensor2figure(tb_pred, slice_indices, colorslist=config.colorslist, is_label=True, fig_title='pred')
            # config.writer.add_figure('figure/{}_batch_seg_temp_pred1_after_resize2originalScale'.format(config_task.task), [img_fig, pred_fig], config.step)

    else:
        raise ValueError('{}: not yet implemented!!'.format(config.unifyPatch))
    
    temp_pred1 = np.asarray(temp_pred1, dtype=np.uint8)

    # for some cases, e.g. Task04_Hippocampus. temp_imgs[0] shape is smaller than model_patch_size.. here use crop to recover to original shape.
    
    if np.any(pad_size):
        temp_pred1 = temp_pred1[pad_size[0]:(original_D+pad_size[0]), pad_size[1]:(original_H+pad_size[1]), pad_size[2]:(original_W+pad_size[2])]

    return temp_pred1


def segment_one_image(config_task, data, model, ID=''):
    """
    perform inference and unpad the volume to original shape
    """
    im = data['image'] # d,h,w,mod
    temp_weight = data['weight'][:,:,:,0] # d,h,w
    original_shape = data['original_shape'] # original_shape, before cropping and resampling
    temp_bbox = data['bbox']
    im_path = data['im_path']
    
    im = im[np.newaxis, ...] # add batch dim

    im2pred = np.transpose(im[0], [3, 0 ,1, 2]) # mod, d, h, w # only one batch? by Chao.
    
    
    st_time = time.time()

    pred1 = batch_segmentation(config_task, im2pred, model)

    logger.info('batch_segmentation time elapsed:{}'.format(tinies.timer(st_time, time.time())))
    
    if config.post_processing:
        st_time = time.time()
        out_label = post_processing(config_task, pred1, temp_weight, ID)
        logger.info('post_processing time elapsed:{}'.format(tinies.timer(st_time, time.time())))

        out_label = np.asarray(out_label, np.int16)
    else:
        out_label = np.asarray(pred1, np.int16)

    st_time = time.time()

    final_label = np.zeros(original_shape, np.int16) # d,h,w
    final_label = set_ND_volume_roi_with_bounding_box_range(config_task, final_label, temp_bbox[0], temp_bbox[1], out_label, sitk.sitkNearestNeighbor, im_path)

    logger.info('set_ND_volume_roi time elapsed:{}'.format(tinies.timer(st_time, time.time())))

    
    return final_label


def multiClassDice(GT, pred, num_class):
    GT = GT.astype(np.int)
    pred = pred.astype(np.int)
    dices = [None]*num_class
    for i in range(num_class):
        pred_tmp = (pred==i).astype(np.int)
        GT_tmp = (GT==i).astype(np.int)
        dices[i] = binary_dice3d(pred_tmp, GT_tmp)
 
    return dices

def evaluate(config_task, ids, model, outdir='eval_out', epoch_num=0):
    """
    evalutation
    """
    files = load_files(ids)
    files = list(files)

    datDir = os.path.join(config.prepData_dir, config_task.task, "Tr")
    dices_list = []
    
    
    # files = files[:2] # debugging.
    logger.info('Evaluating epoch{} for {}--- {} cases:\n{}'.format(epoch_num, config_task.task, len(files), str([obj['id'] for obj in files])))
    for obj in tqdm(files, desc='Eval epoch{}'.format(epoch_num)):
        ID = obj['id']
        # logger.info('evaluating {}:'.format(ID))
        obj['im'] = os.path.join(config.base_dir, config_task.task, "imagesTr", ID)
        obj['gt'] = os.path.join(config.base_dir, config_task.task, "labelsTr", ID)
        img_path = os.path.join(config.base_dir, config_task.task, "imagesTr", ID)
        gt_path = os.path.join(config.base_dir, config_task.task, "labelsTr", ID)
        
        data = get_eval_data(obj, datDir)
        # final_label, probs = segment_one_image(config_task, data, model) # final_label: d, h, w, num_classes
        
        try:
            final_label = segment_one_image(config_task, data, model, ID) # final_label: d, h, w, num_classes
            save_to_nii(final_label, filename=ID + '.nii.gz', refer_file_path=img_path, outdir=outdir, mode="label", prefix='Epoch{}_'.format(epoch_num))

            gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)) # d, h, w
            # treat cancer as organ for Task03_Liver and Task07_Pancreas
            if config_task.task in ['Task03_Liver', 'Task07_Pancreas']:
                gt[gt==2] = 1

            # cal dices
            dices = multiClassDice(gt, final_label, config_task.num_class)
            dices_list.append(dices)
            
            tinies.sureDir(outdir)
            fo = open(os.path.join(outdir,'{}_eval_res.csv'.format(config_task.task)), mode='a+')
            wo = csv.writer(fo, delimiter=',')
            wo.writerow([epoch_num, tinies.datestr(), ID] + dices)
            fo.flush()

            ## for tensorboard visualization
            tb_img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)) # d,h,w
            if tb_img.ndim == 4:
                tb_img = tb_img[0,...]
            train.tb_images([tb_img, gt, final_label], [False, True, True], ['image', 'GT', 'PS'], epoch_num*config.step_per_epoch, tag='Eval_{}_epoch_{}_dices_{}'.format(ID, epoch_num, str(dices)))
        except Exception as e:
            logger.info('{}'.format(str(e)))

    labels = config_task.labels
    dices_all = np.asarray(dices_list)
    dices_mean = dices_all.mean(axis = 0)
    logger.info('Eval mean dices:')
    dices_res = {}
    for i in range(config_task.num_class):
        tag = labels[str(i)]
        dices_res[tag] = dices_mean[i]
        logger.info('    {}, {}'.format(tag, dices_mean[i]))
    
    return dices_res
