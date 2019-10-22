#### @Chao Huang(huangchao09@zju.edu.cn).
import numpy as np

from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug

import ccToolkits.logger as logger
import utils
import tinies

def cc_augment(config_task, data, seg, patch_type, patch_size, patch_center_dist_from_border=30, do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.), do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi), do_scale=True, scale=(0.75, 1.25), border_mode_data='constant', border_cval_data=0, order_data=3, border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1, tag=''):
    # patch_center_dist_from_border should be no more than 1/2 patch size. otherwise code not available.

    # data: [n,c,d,h,w]
    # seg: [n,c,d,h,w]
    dim = len(patch_size)
    
    seg_result = None
    if seg is not None:
        seg_result = np.zeros([seg.shape[0], seg.shape[1]]+patch_size, dtype=np.float32)
    
    data_result = np.zeros([data.shape[0], data.shape[1]] + patch_size, dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]
    
    ## for-loop for dim[0]
    augs = list()
    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)

        # now find a nice center location and extract patch
        if seg is None:
            patch_type = 'any'

        handler = 0
        n = 0
        while handler == 0:

            # augmentation
            modified_coords = False
            if np.random.uniform() < p_el_per_sample and do_elastic_deform:
                a = np.random.uniform(alpha[0], alpha[1])
                s = np.random.uniform(sigma[0], sigma[1])
                coords = elastic_deform_coordinates(coords, a, s)
                modified_coords = True

                augs.append('elastic')

            if np.random.uniform() < p_rot_per_sample and do_rotation:
                if angle_x[0] == angle_x[1]:
                    a_x = angle_x[0]
                else:
                    a_x = np.random.uniform(angle_x[0], angle_x[1])
                if dim == 3:
                    if angle_y[0] == angle_y[1]:
                        a_y = angle_y[0]
                    else:
                        a_y = np.random.uniform(angle_y[0], angle_y[1])
                    if angle_z[0] == angle_z[1]:
                        a_z = angle_z[0]
                    else:
                        a_z = np.random.uniform(angle_z[0], angle_z[1])
                    coords = rotate_coords_3d(coords, a_x, a_y, a_z)
                else:
                    coords = rotate_coords_2d(coords, a_x)
                modified_coords = True

                augs.append('rotation')

            if np.random.uniform() < p_scale_per_sample and do_scale:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])
                coords = scale_coords(coords, sc)
                modified_coords = True

                augs.append('scale')

            # find candidate area for center, the area is cand_point_coord +/- patch_size 
            if patch_type in ['fore', 'small'] and seg is not None:
                if seg.shape[1] > 1:
                    logger.error('TBD for seg with multiple channels')
                if patch_type == 'fore':
                    lab_coords = np.where(seg[sample_id, 0, ...] > 0) # lab_coords: tuple
                elif patch_type == 'small':
                    if config_task.task == 'Task05_Prostate':
                        lab_coords = np.where(seg[sample_id, 0, ...] == 1)
                    else:
                        lab_coords = np.where(seg[sample_id, 0, ...] == config_task.num_class-1)
                if len(lab_coords[0]) > 0: # 0 means no such label exists
                    idx = np.random.choice(len(lab_coords[0]))
                    cand_point_coord = [coords[idx] for coords in lab_coords] # coords for one random point from 'fore' ground
                else:
                    cand_point_coord = None

            if patch_type in ['fore', 'small'] and cand_point_coord is None:
                ctr_list = None
                handler = 1
                data_result = None
                seg_result = None
                augs = None
            else:
                ctr_list = list() # coords of the patch center
                for d in range(dim):
                    if random_crop:
                        if patch_type in ['fore', 'small'] and seg is not None:
                            low = max(patch_center_dist_from_border[d]-1, cand_point_coord[d] - (patch_size[d]/2-1))
                            low = int(low)
                            upper = min(cand_point_coord[d]+(patch_size[d]/2-1), data.shape[d + 2] - (patch_center_dist_from_border[d]-1)) # +/- patch_size[d] is better but computation costly
                            upper = int(upper)

                            if low == upper:
                                ctr = int(low)
                            elif low < upper:
                                ctr = int(np.random.randint(low, upper))
                                # if n > 1:
                                #     logger.info('n:{}; [low,upper]:{}, ctr:{}'.format(n, str([low, upper]), ctr))
                            else:
                                logger.error('(low:{} should be <= upper:{}). patch_type:{}, patch_center_dist_from_border:{}, cand_point_coord:{}, cand point seg value:{}, data.shape:{}, ctr_list:{}'.format(low, upper, str(patch_type), str(patch_center_dist_from_border), str(cand_point_coord), seg[sample_id, 0] + cand_point_coord, str(data.shape), str(ctr_list)))
                        elif patch_type == 'any':
                            if patch_center_dist_from_border[d] == data.shape[d + 2] - patch_center_dist_from_border[d]:
                                ctr = int(patch_center_dist_from_border[d])
                            elif patch_center_dist_from_border[d] < data.shape[d + 2] - patch_center_dist_from_border[d]:
                                ctr = int(np.random.randint(patch_center_dist_from_border[d], data.shape[d + 2] - patch_center_dist_from_border[d]))
                            else:
                                logger.error('low should be <= upper. patch_type:{}, patch_center_dist_from_border:{}, data.shape:{}, ctr_list:{}'.format(str(patch_type), str(patch_center_dist_from_border), str(data.shape), str(ctr_list)))
                    else: # center crop
                        ctr = int(np.round(data.shape[d + 2] / 2.))
                    ctr_list.append(ctr)

                # extracting patch
                if n < 10 and modified_coords:
                    for d in range(dim):
                        coords[d] += ctr_list[d]
                    for channel_id in range(data.shape[1]):
                        data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data, border_mode_data, cval=border_cval_data)
                    if seg is not None:
                        for channel_id in range(seg.shape[1]):
                            seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg, border_mode_seg, cval=border_cval_seg, is_seg=True)
                else:
                    augs = list()
                    if seg is None:
                        s = None
                    else:
                        s = seg[sample_id:sample_id + 1]
                    if random_crop:
                        # margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                        # d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
                        d_tmps = list()
                        for channel_id in range(data.shape[1]):
                            d_tmp = utils.extract_roi_from_volume(data[sample_id, channel_id], ctr_list, patch_size, fill="zero")
                            d_tmps.append(d_tmp)
                        d = np.asarray(d_tmps)
                        if seg is not None:
                            s_tmps = list()
                            for channel_id in range(seg.shape[1]):
                                s_tmp = utils.extract_roi_from_volume(seg[sample_id, channel_id], ctr_list, patch_size, fill="zero")
                                s_tmps.append(s_tmp)
                            s = np.asarray(s_tmps)
                    else:
                        d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
                    # data_result[sample_id] = d[0]
                    data_result[sample_id] = d
                    if seg is not None:
                        # seg_result[sample_id] = s[0]
                        seg_result[sample_id] = s

                ## check patch
                if patch_type in ['fore']: # cancer could be very very small. so use opproximate method (i.e. use 'fore').
                    if np.any(seg_result > 0) and np.any(data_result != 0):
                        handler = 1
                    else:
                        handler = 0
                elif patch_type in ['small']:
                    if config_task.task == 'Task05_Prostate':
                        if np.any(seg_result == 1) and np.any(data_result != 0):
                            handler = 1
                        else:
                            handler = 0
                    else:
                        if np.any(seg_result == config_task.num_class-1) and np.any(data_result != 0):
                            handler = 1
                        else:
                            handler = 0
                else:
                    if np.any(data_result != 0):
                        handler = 1
                    else:
                        handler = 0
                n += 1

                if n > 5:
                    logger.info('tag:{}, patch_type: {}; handler: {}; times: {}; cand point:{}; cand point seg value:{}; ctr_list:{}; data.shape:{}; np.unique(seg_result):{}; np.sum(data_result):{}'.format(tag, patch_type, handler, n, str(cand_point_coord), seg[sample_id, 0, cand_point_coord[0], cand_point_coord[1], cand_point_coord[2]], str(ctr_list), str(data.shape), np.unique(seg_result, return_counts=True), np.sum(data_result)))
                
                
    return data_result, seg_result, augs
