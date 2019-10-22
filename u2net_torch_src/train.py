#### @Chao Huang(huangchao09@zju.edu.cn).
import os
import argparse
import time
import shutil
import itertools
import math
import json
import six
import csv
from multiprocessing import Process, Queue
import multiprocessing
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccToolkits.torchsummary import summary
import ccToolkits.logger as logger

import config
import tinies
import data_utils
from data_utils import (trQueue, get_eval_data)
import evaluate

from loss.lovasz_loss import lovasz_softmax
from loss.dice_loss import MulticlassDiceLoss, one_hot
from loss.focal_loss import FocalLoss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

class tb_load(object):
    '''
    train batch loader for one task
    step1: .enQueue(...)
    step2: .gen(...)
    '''
    def __init__(self, task):
        self.task = task
        self.config_task = config.config_tasks[task]
        super(tb_load,self).__init__()
        self.tr_dataPrep = []
        self.task_archive = None
        self.patch_size = None
        self.nProc = None

    def enQueue(self, task_archive, patch_size):
        ###### prep train data
        self.task_archive = task_archive
        self.patch_size = patch_size
        self.trainQueue = Queue(self.config_task.queue_size) # store patches
        self.nProc = min([self.config_task.nProc, self.config_task.queue_size])

        self.tr_dataPrep = [None] * self.nProc
        for proc in range(self.nProc):
            self.tr_dataPrep[proc] = Process(target=data_utils.trQueue, args=(self.config_task, task_archive['train'], self.trainQueue, self.patch_size, self.nProc, proc))
            self.tr_dataPrep[proc].daemon = True
            self.tr_dataPrep[proc].start()
    def check_process(self):
        procs = list(range(self.nProc))
        # st_time = time.time()
        for i in reversed(procs):
            if self.tr_dataPrep[i].is_alive():
                pass
            else:
                logger.warning('{} Process:{} DIE exitcode: {}'.format(self.task, i, str(self.tr_dataPrep[i].exitcode)))
                # p.close() # new to python 3.7.
                self.tr_dataPrep.remove(self.tr_dataPrep[i])
        del_n = self.nProc - len(self.tr_dataPrep)
        while del_n > 0:
            p = Process(target=data_utils.trQueue, args=(self.config_task, self.task_archive['train'], self.trainQueue, self.patch_size, self.nProc, int(time.time()))) # use int(current time) as seed.
            p.daemon = True
            p.start()
            self.tr_dataPrep.append(p)
            del_n -= 1
        # logger.info('{} time to check_process: {}'.format(self.task, tinies.timer(st_time, time.time())))

    def gen_batch(self, batch_size, patch_size):
        batchImg = np.zeros([batch_size, self.config_task.num_modality, patch_size[0], patch_size[1], patch_size[2]]) # n,mod,d,h,w
        batchLabel = np.zeros([batch_size, patch_size[0], patch_size[1], patch_size[2]]) # n,d,h,w
        batchWeight = np.zeros([batch_size, patch_size[0], patch_size[1], patch_size[2]]) # n,d,h,w
        batchAugs = list()

        # import ipdb; ipdb.set_trace()
        for i in range(batch_size):
            temp_prob = np.random.uniform() 
            st_time = time.time()

            handler = 0
            while handler == 0:
                
                t_wait = 0
                if self.trainQueue.qsize() == 0:
                    logger.info('{} self.trainQueue size = {}, filling....(start time:{})'.format(self.task, self.trainQueue.qsize(), tinies.datestr()))
                while self.trainQueue.qsize() == 0:
                    time.sleep(1)
                    t_wait += 1
                if t_wait > 0:
                    logger.info('{} time to fill self.trainQueue: {}'.format(self.task, t_wait))

                patches = self.trainQueue.get()
                # logger.info('{} trainQueue size:{}'.format(self.task, str(self.trainQueue.qsize())))
                if i <= math.ceil(batch_size/3): # nn_unet3d: at least 1/3 samples in a batch contain at least one forground class
                    if temp_prob < self.config_task.small_prob and patches['small'] is not None:
                        patch = patches['small']
                        handler = 1
                    elif patches['fore'] is not None:
                        patch = patches['fore']
                        handler = 1
                    else:
                        handler = 0
                        logger.warn('handler={}'.format(handler))
                # else for i > math.ceil(batch_size/3)
                else:
                    if temp_prob < self.config_task.small_prob and patches['small'] is not None:
                        patch = patches['small']
                        handler = 1
                    elif 1-temp_prob < self.config_task.fore_prob and patches['fore'] is not None:
                        patch = patches['fore']
                        handler = 1
                    else:
                        patch = patches['any']
                        handler = 1
                if handler == 0:
                    logger.info('handler is 0, going back')
            if handler == 0:
                logger.error('handler is 0')

            # fill in a batch
            batchImg[i,...] = patch['image']
            batchLabel[i,...] = patch['label']
            batchWeight[i,...] = patch['weight']
            batchAugs.append(patch['augs'])

        return (batchImg, batchLabel, batchWeight, batchAugs)
    
    def __len__(self):
        return math.ceil(config.step_per_epoch*config.max_epoch)

def tb_images(array_list, is_label_list, title_list, n_iter, tag=''):
    # tensorboard batch images
    # image: d, h, w
    # pred: d, h, w
    # gt: d, h, w

    colorslist = config.colorslist
    
    slice_indices = config.writer.chooseSlices(array_list[-1], is_label_list[-1]) # arrange the arrays as image1, image2,.., label1, label2,...

    figs = list()
    for i in range(len(array_list)):
        fig = config.writer.tensor2figure(array_list[i], slice_indices, colorslist=colorslist, is_label=is_label_list[i], fig_title=title_list[i])
        figs.append(fig)

    config.writer.add_figure('figure/{}_{}'.format(tag, '_'.join(title_list)), figs, n_iter)


def eval(args, tasks_archive, model, eval_epoch, iterations):
    tasks = args.tasks # list

    model.eval()
    for task_idx in range(len(tasks)):
        config.task_idx = task_idx # needed for u2net3d().
        task = tasks[task_idx]
        config_task = config.config_tasks[task]
        st_time = time.time()

        # evaluating. # tensorboard visualization of eval embedded.
        dices = evaluate.evaluate(config_task, tasks_archive[task]['fold' + str(args.fold)]['val'], model, epoch_num=eval_epoch, outdir=config.eval_out_dir)
    
        fo = open(os.path.join(config.eval_out_dir,'{}_eval_res.csv'.format(args.trainMode)), mode='a+')
        wo = csv.writer(fo, delimiter=',')
        for k, v in dices.items():
            config.writer.add_scalar('data/dices/{}_{}'.format(task, k), v, iterations)
            wo.writerow([args.trainMode, task, eval_epoch, config.step_per_epoch, k, v, tinies.datestr()])
        fo.flush()
        logger.info('Eval time elapsed:{}'.format(tinies.timer(st_time, time.time())))


def train(args, tasks_archive, model):
    torch.backends.cudnn.benchmark=True
    
    if args.resume_ckp != '':
        logger.info('==> loading checkpoint: {}'.format(args.ckp))
        checkpoint = torch.load(args.resume_ckp)

    model = nn.parallel.DataParallel(model)
    
    logger.info('  + model num_params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if config.use_gpu:
        model.cuda() # required bofore optimizer?
    #     cudnn.benchmark = True

    print(model) # especially useful for debugging model structure.
    # summary(model, input_size=tuple([config.num_modality]+config.patch_size)) # takes some time. comment during debugging. ouput each layer's out shape.
    # for name, m in model.named_modules():
    #     logger.info('module name:{}'.format(name))
    #     print(m)

    # lr
    lr = config.base_lr
    if args.resume_ckp != '':
        optimizer = checkpoint['optimizer']
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay) # 
        
    # loss
    dice_loss = MulticlassDiceLoss()
    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2)

    # prep data
    tasks = args.tasks # list
    tb_loaders = list() # train batch loader
    len_loader = list()
    for task in tasks:
        tb_loader = tb_load(task)
        tb_loader.enQueue(tasks_archive[task]['fold' + str(args.fold)], config.patch_size)
        tb_loaders.append(tb_loader)
        len_loader.append(len(tb_loader))
    min_len_loader = np.min(len_loader)

    
    # init train values
    if args.resume_ckp != '':
        trLoss_queue = checkpoint['trLoss_queue']
        last_trLoss_ma = checkpoint['last_trLoss_ma']
    else:
        trLoss_queue = deque(maxlen=config.trLoss_win) # queue to store exponential moving average of total loss in last N epochs
        last_trLoss_ma = None # the previous one.
    trLoss_queue_list = [deque(maxlen=config.trLoss_win) for i in range(len(tasks))]
    last_trLoss_ma_list = [None] * len(tasks)
    trLoss_ma_list = [None] * len(tasks)
   
    if args.resume_epoch > 0:
        start_epoch = args.resume_epoch + 1
        iterations = args.resume_epoch*config.step_per_epoch + 1
    else: 
        start_epoch = 1
        iterations = 1
    logger.info('start epoch: {}'.format(start_epoch))

    ## run train
    for epoch in range(start_epoch, config.max_epoch+1):
        logger.info('    ----- training epoch {} -----'.format(epoch))
        epoch_st_time = time.time()
        model.train()
        loss_epoch = 0.0
        loss_epoch_list = [0] * len(tasks)
        num_batch_processed = 0 # growing
        num_batch_processed_list = [0] * len(tasks)
        
        for step in tqdm(range(config.step_per_epoch), desc='{}: epoch{}'.format(args.trainMode, epoch)):
            config.step = iterations
            config.task_idx = (iterations-1) % len(tasks)
            config.task = tasks[config.task_idx]
            # import ipdb; ipdb.set_trace()

            # tb show lr
            config.writer.add_scalar('data/lr', lr, iterations-1)

            st_time = time.time()
            for idx in range(len(tasks)):
                tb_loaders[idx].check_process()
            # import ipdb; ipdb.set_trace()
            (batchImg, batchLabel, batchWeight, batchAugs) = tb_loaders[config.task_idx].gen_batch(config.batch_size, config.patch_size)
            # logger.info('idx{}_{}, gen_batch time elapsed:{}'.format(config.task_idx, config.task, tinies.timer(st_time, time.time())))

            st_time = time.time()
            batchImg = torch.from_numpy(batchImg).float() # change all inputs to same torch tensor type
            batchLabel = torch.from_numpy(batchLabel).float()
            batchWeight = torch.from_numpy(batchWeight).float()
            
            if config.use_gpu:
                batchImg = batchImg.cuda()
                batchLabel = batchLabel.cuda()
                batchWeight = batchWeight.cuda()
            # logger.info('idx{}_{}, .cuda time elapsed:{}'.format(config.task_idx, config.task, tinies.timer(st_time, time.time())))

            optimizer.zero_grad()

            st_time = time.time()
            if config.trainMode in ["universal"]:
                output, share_map, para_map = model(batchImg)
            else:
                output = model(batchImg)
            # logger.info('idx{}_{}, model() time elapsed:{}'.format(config.task_idx, config.task, tinies.timer(st_time, time.time())))

            st_time = time.time()
            # tensorboard visualization of training
            for i in range(len(tasks)):
                if iterations > 200 and iterations % 1000 == i:
                    tb_images([batchImg[0,0,...], batchLabel[0,...], torch.argmax(output[0,...], dim=0)], [False, True, True], ['image', 'GT', 'PS'], iterations, tag='Train_idx{}_{}_batch{}_{}'.format(config.task_idx, config.task, 0, '_'.join(batchAugs[0])))

                    tb_images([batchImg[config.batch_size-1,0,...], batchLabel[config.batch_size-1,...], torch.argmax(output[config.batch_size-1,...], dim=0)], [False, True, True], ['image', 'GT', 'PS'], iterations, tag='Train_idx{}_{}_batch{}_{}_step{}'.format(config.task_idx, config.task, config.batch_size-1, '_'.join(batchAugs[config.batch_size-1]), iterations-1))
                    if config.trainMode == "universal":
                        logger.info('share_map shape:{}, para_map shape:{}'.format(str(share_map.shape), str(para_map.shape)))
                        tb_images([para_map[0,:,64,...], share_map[0,:,64,...]], [False, False], ['last_para_map', 'last_share_map'], iterations, tag='Train_idx{}_{}_para_share_maps_channels'.format(config.task_idx, config.task))
                    
            logger.info('----- {}, train epoch {} time elapsed:{} -----'.format(config.task, epoch, tinies.timer(epoch_st_time, time.time())))

            st_time = time.time()

            output_softmax = F.softmax(output, dim=1)
            
            loss = lovasz_softmax(output_softmax, batchLabel, ignore=10) + focal_loss(output, batchLabel)

            loss.backward()
            optimizer.step()

            # logger.info('idx{}_{}, backward time elapsed:{}'.format(config.task_idx, config.task, tinies.timer(st_time, time.time())))
            
            # loss.data.item()
            config.writer.add_scalar('data/loss_step', loss.item(), iterations)
            config.writer.add_scalar('data/loss_step_idx{}_{}'.format(config.task_idx, config.task), loss.item(), iterations)

            loss_epoch += loss.item()
            num_batch_processed += 1

            loss_epoch_list[config.task_idx] += loss.item()
            num_batch_processed_list[config.task_idx] += 1

            iterations +=1

        # import ipdb; ipdb.set_trace()
        if epoch % config.save_epoch == 0:
            ckp_path = os.path.join(config.log_dir, '{}_{}_epoch{}_{}.pth.tar'.format(args.trainMode, '_'.join(args.tasks), epoch, tinies.datestr()))
            torch.save({
                'epoch': epoch,
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'trLoss_queue': trLoss_queue,
                'last_trLoss_ma': last_trLoss_ma
            }, ckp_path)

        loss_epoch /= num_batch_processed

        config.writer.add_scalar('data/loss_epoch', loss_epoch, iterations-1)
        for idx in range(len(tasks)):
            task = tasks[idx]
            loss_epoch_list[idx] /= num_batch_processed_list[idx]
            config.writer.add_scalar('data/loss_epoch_idx{}_{}'.format(idx, task), loss_epoch_list[idx], iterations-1)
        # import ipdb; ipdb.set_trace()
        
        ### lr decay
        trLoss_queue.append(loss_epoch)
        trLoss_ma = np.asarray(trLoss_queue).mean() # moving average. What about exponential moving average
        config.writer.add_scalar('data/trLoss_ma', trLoss_ma, iterations-1)

        for idx in range(len(tasks)):
            task = tasks[idx]
            trLoss_queue_list[idx].append(loss_epoch_list[idx])
            trLoss_ma_list[idx] = np.asarray(trLoss_queue_list[idx]).mean() # moving average. What about exponential moving average
            config.writer.add_scalar('data/trLoss_ma_idx{}_{}'.format(idx, task), trLoss_ma_list[idx], iterations-1)

        # import ipdb; ipdb.set_trace()
        #### online eval
        Eval_bool = False
        if epoch >= config.start_val_epoch and epoch % config.val_epoch == 0:
            Eval_bool = True
        elif lr < 1e-8:
            Eval_bool = True
            logger.info('lr is reduced to {}. Will do the last evaluation for all samples!'.format(lr))
        
        else:
            pass
        # if epoch >= config.start_val_epoch and epoch % config.val_epoch == 0:
        if Eval_bool:
            eval(args, tasks_archive, model, epoch, iterations-1)

           
        ## stop if lr is too low
        if lr < 1e-8:
            logger.info('lr is reduced to {}. Job Done!'.format(lr))
            break
        

        ###### lr decay based on current task
        if len(trLoss_queue) == trLoss_queue.maxlen:
            if last_trLoss_ma and last_trLoss_ma - trLoss_ma < 1e-4: # 5e-3
                lr /= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            last_trLoss_ma = trLoss_ma

        ## save model when lr < 1e-8
        if lr < 1e-8:
            ckp_path = os.path.join(config.log_dir, '{}_{}_epoch{}_{}.pth.tar'.format(args.trainMode, '_'.join(args.tasks), epoch, tinies.datestr()))
            torch.save({
                'epoch': epoch,
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'trLoss_queue': trLoss_queue,
                'last_trLoss_ma': last_trLoss_ma
            }, ckp_path)
        
        
         