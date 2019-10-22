#### @Chao Huang(huangchao09@zju.edu.cn).
# main file to train a universal model with the tasks in round robin fashion.
# CUDA_VISIBLE_DEVICES=2 python train_model_wt_adapters.py --trainMode 'universal' --module 'separable_adapter' --out_tag '20190213' --tasks Task02_Heart' 'Task03_Liver' 'Task04_Hippocampus' 'Task05_Prostate' 'Task07_Pancreas'

import os
import time
import argparse
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from ccToolkits.MySummaryWriter import MySummaryWriter
from ccToolkits import logger

import config
import tinies
import train
from models import u2net3d

parser = argparse.ArgumentParser(description='u2net')
parser.add_argument('--tasks', default='Task02_Heart', nargs='+', help='Task(s) to be trained') # e.g. python try1try.py --dataset 'a' 'b' 'c'
parser.add_argument('--trainMode', default='universal', type=str, help='Task adaptation mode') # independent, shared, universal. shared: used to train a unet3d model w/o adapters for all tasks
parser.add_argument('--module', default='adapter', type=str, help='specific module type: series_adapter, parallel_adapter, separable_adapter')
parser.add_argument('--ckp', default='', type=str, help='dir to load ckp for transfer learning')
parser.add_argument('--resume_ckp', default='', type=str, help='dir to load ckp for evaluation or training')
parser.add_argument('--resume_epoch', default=0, type=int, help='epoch of resume_ckp')
parser.add_argument('--fold', default=0, type=int, help='fold index')
parser.add_argument('--model', help='model name', default='u2net3d')
# parser.add_argument('--loss', default='dice_loss', nargs='+', help='loss funcs: i.e. dice, ce, lovasz, focal') # e.g. python try1try.py --loss 'a' 'b' 'c'
parser.add_argument('--out_tag', default='', type=str, help='output dir tag')
parser.add_argument('--base_outChans', default=32, type=int, help='base_outChans')
parser.add_argument('--predict',  default='True', type=str, help="Run prediction")
parser.add_argument('--pred_epoch', default=120, type=int, help="Run prediction with ckp of epoch num")
args = parser.parse_args()

tinies.sureDir(config.prepData_dir)

if config.use_gpu and torch.cuda.is_available():
    config.use_gpu = True
else:
    config.use_gpu = False

config.trainMode = args.trainMode
config.module = args.module
if args.module == 'separable_adapter':
    config.base_outChans = args.base_outChans 
else:
    pass

#### config tasks
if type(args.tasks) is str:
    args.tasks = [args.tasks]
# args.tasks = sorted(args.tasks)
for task in args.tasks:
    config.config_tasks[task] = config.set_config_task(args.trainMode, task, config.base_dir)

if args.out_tag:
    args.out_tag = '_'+args.out_tag

#### Prepare datasets
with open(os.path.join(os.path.dirname(os.getcwd()), 'fold_splits.json'), mode='r') as f:
    tasks_archive = json.load(f) # dict: {'Task02_Heart'/...}{'fold index'}{'train'/'val'}

# seed
np.random.seed(1993)

#### prep train
if args.trainMode == "independent":
    logger.error('trainMode should be one of parallel_adapter, shared_adapter')
        
elif args.trainMode != "independent":
    ### model settings
    config.patch_size = [128,128,128]
    config.patch_weights = tinies.calPatchWeights(config.patch_size)
    
    config.out_dir = os.path.join(config.out_dir, 'res_{}_{}{}'.format(args.model, args.trainMode, args.out_tag), '_'.join(args.tasks))
    tinies.sureDir(config.out_dir)
    config.eval_out_dir = os.path.join(config.out_dir, "eval_out")
    tinies.newdir(config.eval_out_dir)

    config.log_dir = os.path.join(config.out_dir, 'train_log')
    config.writer = MySummaryWriter(log_dir=config.log_dir) # this will create log_dir
    logger.set_logger_dir(os.path.join(config.log_dir, 'logger'), action="b") # 'b' reuse log_dir and backup log.log
    logger.info('--------------------------------Training for {}: {}--------------------------------'.format(args.trainMode, '_'.join(args.tasks)))

    # instantialize model
    inChans_list = [config.config_tasks[task].num_modality for task in args.tasks] # input num_modality
    num_class_list = [config.config_tasks[task].num_class for task in args.tasks]
    model = u2net3d.u2net3d(inChans_list=inChans_list, base_outChans=config.base_outChans, num_class_list=num_class_list)
    torch.manual_seed(1)
    model.apply(train.weights_init)
    
    # if transfer learning
    # Load checkpoint and initialize the networks with the weights of a pretrained network
    if args.ckp != '' and args.resume_ckp == '':
        logger.info('==> Transferring from checkpoint: {}, loading checkpoint.....'.format(args.ckp))
        checkpoint = torch.load(args.ckp)
        model_old = checkpoint['model']

        shared_modules = ['down_blocks', 'up_blocks'] # cautions: no residual block in , 'down_samps', 'up_samps'
        # task specific module names are marked with "*.adap*"
        # store and copy shared conv3x3, conv1x1, instancenorm to new neural network.

        # import ipdb; ipdb.set_trace()

        if args.trainMode != 'independent':
            logger.info('==> Transferring from checkpoint: {}, transfering shared weights.....'.format(args.ckp))
            store_weight3x3 = []
            store_weight1x1 = []
            store_weightNorm = []

            name3x3 = []
            name1x1 = []
            nameNorm = []

            for name, m in model_old.named_modules():
                if any(i in name for i in shared_modules):
                    if '.adap' not in name and isinstance(m, nn.Conv3d) and (m.kernel_size[0]==3):
                        store_weight3x3.append(m.weight.data)
                        name3x3.append(name)
                    elif '.adap' not in name and isinstance(m, nn.Conv3d) and (m.kernel_size[0]==1):
                        store_weight1x1.append(m.weight.data)
                        name1x1.append(name)
                    elif '.adap' not in name and isinstance(m, nn.InstanceNorm3d):
                        store_weightNorm.append(m.weight.data)
                        nameNorm.append(name)
            
            element3x3 = 0
            element1x1 = 0
            elementNorm = 0
            for name, m in model.named_modules():
                if any(i in name for i in shared_modules):
                    if '.adap' not in name and isinstance(m, nn.Conv3d) and (m.kernel_size[0]==3):
                        m.weight.data = store_weight3x3[element3x3]
                        m.weight.requires_grad = False # # Freeze shared weights
                        element3x3 += 1
                    elif '.adap' not in name and isinstance(m, nn.Conv3d) and (m.kernel_size[0]==1):
                        m.weight.data = store_weight1x1[element1x1]
                        m.weight.requires_grad = False # # Freeze shared weights
                        element1x1 += 1
                    
        else:
            logger.error('TBD for adapters other than parallel_adapter')
    else:
        logger.info('------------------------------------------------------------------')
        logger.info('------------  train model with adapters from scratch -------------')
        logger.info('------------------------------------------------------------------')
    
    # if resume
    if args.resume_ckp != '':
        checkpoint = torch.load(args.resume_ckp)
        model = checkpoint['model']

    train.train(args, tasks_archive, model)

config.writer.close # close tensorboardX summarywriter