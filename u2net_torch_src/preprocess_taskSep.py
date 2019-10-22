#### @Chao Huang(huangchao09@zju.edu.cn).
import os
import shutil
import argparse
import json
import time

from glob2 import glob
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import multiprocessing

import tinies
import config
import utils


tinies.sureDir(config.prepData_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nProc', type=int, default=8, help='process workers to create')
    args = parser.parse_args()

    ################################################################
    ### resample and crop et al.
    # tasks = sorted([x for x in os.listdir(config.base_dir) if x.startswith('Task')])
    tasks = ['Task02_Heart', 'Task03_Liver', 'Task04_Hippocampus', 'Task05_Prostate', 'Task07_Pancreas', 'Task09_Spleen'] # 
    for task in tqdm(tasks):
        # task = 'Task04_Hippocampus'
        print(task)
        # task_archive = tasks_archive[task]
        config_task = config.set_config_task('independent', task, config.base_dir)

        def prep(files, outDir, with_gt=True):
            print("ids[0]:{}, current time:{}".format(os.path.basename(files[0]), str(tinies.datestr())))
            for img_path in files:
                # tinies.ForkedPdb().set_trace()
                ID=os.path.basename(img_path).split('.')[0]
                if with_gt:
                    lab_path = os.path.join(config.base_dir, task, 'labelsTr', ID)
                else:
                    lab_path = None
                volume_list, label, weight, original_shape, [bbmin, bbmax] = utils.preprocess(img_path, lab_path, config_task, with_gt=with_gt)
                volumes = np.asarray(volume_list)
                np.save(os.path.join(outDir, ID+'_volumes.npy'), volumes)
                if with_gt:
                    np.save(os.path.join(outDir, ID+'_label.npy'), label)
                np.save(os.path.join(outDir, ID+'_weight.npy'), weight)

                json_info = dict()
                json_info['original_shape'] = str(original_shape) # use eval() to unstr
                json_info['bbox'] = str([bbmin, bbmax]) # use eval() to unstr
                with open(os.path.join(outDir, ID+'.json'), 'w') as f:
                    json.dump(json_info, f, indent=4)

        tr_files = sorted([x for x in glob(os.path.join(config.base_dir, task, 'imagesTr', '*')) if '.nii.gz' in x])
        trDir = os.path.join(config.prepData_dir, task, "Tr")
        tinies.sureDir(trDir) # make dir if not existing

        ts_files = sorted([x for x in glob(os.path.join(config.base_dir, task, 'imagesTs', '*')) if '.nii.gz' in x])
        tsDir = os.path.join(config.prepData_dir, task, "Ts")
        tinies.sureDir(tsDir) # make dir if not existing

        pool = multiprocessing.Pool(args.nProc) # processes=3
        pool.apply_async(func=prep, args=(tr_files, trDir, True))
        pool.apply_async(func=prep, args=(ts_files, tsDir, False))
        pool.close() # close pool, no more processes added to pool
        pool.join() # wait pool to finish, required and should be after .close()


        ######################################################################################
        ## fuse cancer to organ
        ## fuse 'cancer' to Liver for Task03_Liver or to Pancreas for Task07_Pancreas
        ## the original .npy data is copied to xxx_with_cancer_as_2
        # tasks = sorted([x for x in os.listdir(config.base_dir) if x.startswith('Task')])
        fuseCa = True
        if fuseCa:
            print('Fusing cancer to organ...')
            tasks = ['Task03_Liver', 'Task07_Pancreas']
            for task in tqdm(tasks):
                # task = 'Task03_Liver'
                print(task)
                # task_archive = tasks_archive[task]
                config_task = config.set_config_task('independent', task, config.base_dir)

                def fuse(files, outDir, with_gt=True):
                    print("ids[0]:{}, current time:{}".format(os.path.basename(files[0]), str(tinies.datestr())))
                    for lab_path in files:
                        print('loading:{}'.format(lab_path))
                        # tinies.ForkedPdb().set_trace()
                        label = np.load(lab_path)
                        label[label == 2] = 1 # cancer fused to organ
                        np.save(os.path.join(lab_path), label)
                
                task_prep_dir = os.path.join(config.prepData_dir, task)
                files = sorted([x for x in glob(os.path.join(task_prep_dir, 'Tr', '*')) if '_label.npy' in x])
                outDir = os.path.join(task_prep_dir, "Tr")

                pool = multiprocessing.Pool(args.nProc) # processes=3
                pool.apply_async(func=fuse, args=(files, outDir, True))
                pool.close() # close pool, no more processes added to pool
                pool.join() # wait pool to finish, required and should be after .close()
        else:
            print('NOT fusing cancer to organ!')
            pass


