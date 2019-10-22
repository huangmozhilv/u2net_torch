#### @Chao Huang(huangchao09@zju.edu.cn).
import os
import json

import numpy as np

from ccToolkits.MySummaryWriter import MySummaryWriter

# global placeholder
writer = None 

### data path
prepData_dir = ['../prepData/']
base_dir = ['../dataset/']
out_dir = "../results"

# placeholders for debugging or universal model
eval_out_dir = None
test_out_dir = None
log_dir = None
patch_size = [128,128,128]
unifyPatch = 'resize' # 'resize', resize task specific patch size to same patch size for all tasks; 'pad', pad task specific patch size to same patch size for all tasks
patch_weights = None
batch_size = 2
num_pool_per_axis = [5,5,5]
base_outChans = 16
step = 0 # current step
start_val_epoch = 100
val_epoch = 20 # val every 10 epochs.

# training config
trainMode = None # placeholder. choices: independent, shared, universal
module = 'separable_adapter' # specific module type for universal model: series_adapter, parallel_adapter, separable_adapter
step_per_epoch = 300 # iterations per epoch
val_epochs = {'Task02_Heart':10, 'Task03_Liver':10, 'Task04_Hippocampus':10, 'Task05_Prostate':10, 'Task07_Pancreas':10, 'Task09_Spleen':10}
# 'Task02_Heart' 'Task03_Liver' 'Task04_Hippocampus' 'Task05_Prostate' 'Task07_Pancreas' 'Task09_Spleen'
max_epoch = 300 # max training epochs
save_epoch = 5

trLoss_win = 20
base_lr = 0.0003
weight_decay = 5e-4 # to Adam, l2 penalty # prevent overfitting

### tensorboard visualization
colorslist=['#000000','#00FF00','#0000FF','#FF0000', '#FFFF00']

### model config
task_idx = 0 # used for universal model.
task = 'Task02_Heart'
use_gpu = True
residual = True
post_processing = True
deep_supervision = True
instance_norm = True
batch_norm = False
data_sampling = 'one_third_positive' # all_positive, random, one_positive, one_third_positive
intensity_norm = 'modality' # different norm method
test_flip = False # Test time augmentation

#### task-specific settings
batch_sizes = {'Task02_Heart':2, 'Task03_Liver':2, 'Task04_Hippocampus':9, 'Task05_Prostate':4, 'Task07_Pancreas':2, 'Task09_Spleen':2}

base_outChanss = {'Task02_Heart':16, 'Task03_Liver':16, 'Task04_Hippocampus':16, 'Task05_Prostate':16, 'Task07_Pancreas':16, 'Task09_Spleen':16}

patch_sizes = {'Task02_Heart':[80,192,128], 'Task03_Liver':[128,128,128], 'Task04_Hippocampus':[40,56,40], 'Task05_Prostate':[20,192,192], 'Task07_Pancreas':[96,160,128], 'Task09_Spleen':[96,160,128]}

nums_pool_per_axis = {'Task02_Heart':[4,5,5], 'Task03_Liver':[5,5,5], 'Task04_Hippocampus':[3,3,3], 'Task05_Prostate':[2,5,5], 'Task07_Pancreas':[4,5,5], 'Task09_Spleen':[4,5,5]}

pixel_spacings = {'Task02_Heart':[1.37,1.25,1.25], 'Task03_Liver':[1,0.77,0.77], 'Task04_Hippocampus':[1,1,1], 'Task05_Prostate':[3.6,0.625,0.625], 'Task07_Pancreas':[2.5,0.8,0.8], 'Task09_Spleen':[5,0.8,0.8]} # 


# data augmentation
# fixed nProcs_ind: T03, 
nProcs_ind = {'Task02_Heart':6, 'Task03_Liver':20, 'Task04_Hippocampus':5, 'Task05_Prostate':8, 'Task07_Pancreas':15, 'Task09_Spleen':10} # number of processors
nProcs_uni = {'Task02_Heart':4, 'Task03_Liver':15, 'Task04_Hippocampus':2, 'Task05_Prostate':3, 'Task07_Pancreas':4, 'Task09_Spleen':5}
qsizes_ind = {'Task02_Heart':5, 'Task03_Liver':8, 'Task04_Hippocampus':10, 'Task05_Prostate':5, 'Task07_Pancreas':8, 'Task09_Spleen':5} # queue_sizes
qsizes_uni = {'Task02_Heart':5, 'Task03_Liver':6, 'Task04_Hippocampus':4, 'Task05_Prostate':5,'Task07_Pancreas':6, 'Task09_Spleen':5}
num_patch_per_files = {'Task02_Heart':2, 'Task03_Liver':2, 'Task04_Hippocampus':2, 'Task05_Prostate':2, 'Task07_Pancreas':2, 'Task09_Spleen':2}

scale_probs = {'Task02_Heart':0.5, 'Task03_Liver':0.5, 'Task04_Hippocampus':0.5, 'Task05_Prostate':0.5, 'Task07_Pancreas':0.5, 'Task09_Spleen':0.5} # TBD

small_probs = {'Task02_Heart':0.3, 'Task03_Liver':0.3, 'Task04_Hippocampus':0.3, 'Task05_Prostate':0.3, 'Task07_Pancreas':0.3, 'Task09_Spleen':0} # oversample smallest objects.
fore_probs = {'Task02_Heart':0.3, 'Task03_Liver':0.3, 'Task04_Hippocampus':0.3, 'Task05_Prostate':0.3, 'Task07_Pancreas':0.3, 'Task09_Spleen':0} # TBD

deform_probs = {'Task02_Heart':0.3, 'Task03_Liver':0.3, 'Task04_Hippocampus':0.3, 'Task05_Prostate':0.3, 'Task07_Pancreas':0.3, 'Task09_Spleen':0.3}
rotate_probs = {'Task02_Heart':0.3, 'Task03_Liver':0.3, 'Task04_Hippocampus':0.3, 'Task05_Prostate':0.3, 'Task07_Pancreas':0.3, 'Task09_Spleen':0.3}
mirror_probs = {'Task02_Heart':0.3, 'Task03_Liver':0.3, 'Task04_Hippocampus':0.3, 'Task05_Prostate':0.3, 'Task07_Pancreas':0.3, 'Task09_Spleen':0.3}

### set config for each task
tasks = ['Task02_Heart', 'Task03_Liver', 'Task04_Hippocampus', 'Task05_Prostate', 'Task07_Pancreas', 'Task09_Spleen']
task = "Task04_Hippocampus" # placeholder. for debugging

config_tasks = dict() # placeholder
class set_config_task(object):
    '''
    set config for each task: 'Task02_Heart'/..../'Task9_Spleen'
    '''
    def __init__(self, trainMode, task, base_dir):
        self.task = task
        self.pixel_spacing = pixel_spacings[task] # no init

        with open(os.path.join(base_dir, task, 'dataset.json'), mode='r', encoding="utf-8") as f:
            print('file path is :{}'.format(os.path.join(base_dir, task, 'dataset.json')))
            task_info = json.load(f)
            if task in ['Task03_Liver', 'Task07_Pancreas']:
                keys = list(task_info['labels'].keys())
                for key in keys:
                    if task_info['labels'][key] in ['cancer', 'tumour']:
                        task_info['labels'].pop(key)
                self.labels = task_info['labels']
                print('{} labels:{}'.format(task, str(self.labels.keys())))
                self.num_class = len(self.labels)
            else:
                self.labels = task_info['labels'] # no init
                self.num_class = len(self.labels) # len(task_info['labels'])
            self.modality = task_info['modality']
            self.num_modality = len(task_info['modality'])
            
        # model settings
        self.patch_size = patch_sizes[task]

        # data prep
        if trainMode in ['independent']:
            self.nProc = nProcs_ind[task]
            self.queue_size = qsizes_ind[task]
        elif trainMode in ['shared', 'universal']:
            self.nProc = nProcs_uni[task]
            self.queue_size = qsizes_uni[task]
        # self.queue_size = queue_sizes[task]
        self.num_patch_per_file = num_patch_per_files[task]
        # data augmentation settings
        self.scale_prob = scale_probs[task]
        self.small_prob = small_probs[task] # at least one smallest label
        self.fore_prob = fore_probs[task] # at least one any foreground label
        self.deform_prob = deform_probs[task]
        self.rotate_prob = rotate_probs[task]
        self.mirror_prob = mirror_probs[task]