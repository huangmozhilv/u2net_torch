#### @Chao Huang(huangchao09@zju.edu.cn).
import os
import time
import argparse
import csv
import json

import SimpleITK as sitk
import numpy as np
import pandas as pd
from glob2 import glob
from tqdm import tqdm

import config

parser = argparse.ArgumentParser(description='msd')
parser.add_argument('--task', type=str, default='meta') # meta, extract meta info; 
parser.add_argument('--root', type=str, default='/Users/messi/Documents/PythonProjects', help='path to save msd_meta.csv')
args = parser.parse_args()

if args.task == 'meta':
    # taskF = 'Task02_Heart'
    tasks = [i for i in os.listdir(config.base_dir) if '.DS_Store' not in i]
    tasks = sorted(tasks)
    meta = dict()
    meta['task'] = list()
    meta['patID'] = list()

    print('task..., patID...')
    for taskF in tqdm(tasks):
        meta['task'].extend([taskF for f in glob(os.path.join(config.base_dir, taskF, 'imagesTr', '*.nii.gz'))])
        meta['patID'].extend([os.path.basename(f).split('.')[0] for f in glob(os.path.join(config.base_dir, taskF, 'imagesTr', '*.nii.gz'))])

    # meta
    meta['s_lab_voxs'] = list()
    meta['s_lab_percent'] = list()
    for key in tqdm(['bitpix', 'datatype', 'dim[0]', 'dim[1]', 'dim[2]', 'dim[3]', 'dim[4]', 'dim[5]', 'dim[6]', 'dim[7]', 'dim_info', 'pixdim[0]', 'pixdim[1]', 'pixdim[2]', 'pixdim[3]', 'pixdim[4]', 'pixdim[5]', 'pixdim[6]', 'pixdim[7]', 'scl_inter', 'scl_slope', 'srow_x', 'srow_y', 'srow_z']):
        meta[key] = list()
    for taskF in tqdm(tasks, desc='task'):
        with open(os.path.join(config.base_dir, taskF, 'dataset.json'), mode='r') as f:
            task_info = json.load(f)
            num_class = len(task_info['labels'])
        for f in tqdm(glob(os.path.join(config.base_dir, taskF, 'imagesTr', '*.nii.gz')), desc='f_path'):
            ID = os.path.basename(f).split('.')[0]
            # img meta
            sitk_img = sitk.ReadImage(f)
            for key in ['bitpix', 'datatype', 'dim[0]', 'dim[1]', 'dim[2]', 'dim[3]', 'dim[4]', 'dim[5]', 'dim[6]', 'dim[7]', 'dim_info', 'pixdim[0]', 'pixdim[1]', 'pixdim[2]', 'pixdim[3]', 'pixdim[4]', 'pixdim[5]', 'pixdim[6]', 'pixdim[7]', 'scl_inter', 'scl_slope', 'srow_x', 'srow_y', 'srow_z']: # part of sitk_image.GetMetaDataKeys()
                meta[key].append(sitk_img.GetMetaData(key))
            
            # lab
            
            lab_path = os.path.join(config.base_dir, taskF, 'labelsTr', ID+'.nii.gz')
            lab = sitk.GetArrayFromImage(sitk.ReadImage(lab_path))
            lab[lab>10] = 0 # Task04_Hippocampus 003 and 243 have one wrong gt pixel assigned 254. Here arbitrarily set to 0 （background). By Chao.
            if taskF == 'Task05_Prostate':
                s_lab_voxs = np.sum(lab==1)
            else:
                s_lab_voxs = np.sum(lab==(num_class-1))
            meta['s_lab_voxs'].append(s_lab_voxs)
            meta['s_lab_percent'].append(s_lab_voxs/lab.size)

    with open(os.path.join(os.getcwd(), 'msd_meta.csv'), 'w', newline='') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(meta.keys())
        for i in range(len(meta['task'])):
            w.writerow([meta[key][i] for key in meta.keys()])

elif 'stat' in args.task.lower():
    root = args.root
    # root = '/Users/messi/Documents/PythonProjects'
    meta_file = os.path.join(root, 'msd_meta.csv')
    stat_file = os.path.join(root, 'msd_stat.csv')
    msd_meta = pd.read_csv(meta_file)
    msd_meta.columns
    # msd_meta.loc[msd_meta.task == 'Task02_Heart','dim[1]'].mean()
    tasks = list(msd_meta.task.unique())

    df = pd.DataFrame(np.nan, index=range(len(tasks)), columns=['task', 'dim[1]_min', 'dim[1]_max', 'dim[1]_mean', 'dim[1]_median', 'dim[2]_min', 'dim[2]_max', 'dim[2]_mean', 'dim[2]_median', 'dim[3]_min', 'dim[3]_max', 'dim[3]_mean', 'dim[3]_median', 'pixdim[1]_min', 'pixdim[1]_max', 'pixdim[1]_mean', 'pixdim[1]_median', 'pixdim[2]_min', 'pixdim[2]_max', 'pixdim[2]_mean', 'pixdim[2]_median', 'pixdim[3]_min', 'pixdim[3]_max', 'pixdim[3]_mean', 'pixdim[3]_median', 's_lab_voxs_min', 's_lab_voxs_max', 's_lab_voxs_mean', 's_lab_voxs_median', 's_lab_percent_min', 's_lab_percent_max', 's_lab_percent_mean', 's_lab_percent_median'])

    for i in range(len(tasks)):
        task = tasks[i]
        df.loc[i, 'task'] = task
        for col in ['dim[1]', 'dim[2]', 'dim[3]', 'pixdim[1]', 'pixdim[2]', 'pixdim[3]', 's_lab_voxs', 's_lab_percent']:
            df.loc[i, col+'_min'] = msd_meta.loc[msd_meta.task == task, col].min()
            df.loc[i, col+'_max'] = msd_meta.loc[msd_meta.task == task, col].max()
            df.loc[i, col+'_mean'] = msd_meta.loc[msd_meta.task == task, col].mean()
            df.loc[i, col+'_median'] = msd_meta.loc[msd_meta.task == task, col].median()

    df.to_csv(stat_file, index=False)