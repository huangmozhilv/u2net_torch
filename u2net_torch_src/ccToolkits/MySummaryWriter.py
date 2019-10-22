#### @Chao Huang(huangchao09@zju.edu.cn).
import math

import numpy as np

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from matplotlib import colors
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import ccToolkits.logger as logger

colorslist = ['#000000','#00FF00','#0000FF','#FF0000', '#FFFF00'] # hex color: Black, Green, Blue, Red, Yellow


def selectSlices(array, is_label=True):
    # numpy array. d, h, w
    # return target z_list()
    z = array.shape[0]
    if not is_label:
        # return [int(i) for i in np.random.choice(z, 1).tolist()] # a random slice could be all 0 as it was padded with 0.
        step = math.ceil(z/8) # will result in 7 or 8 points
        out = list(range(0, z, step))
    elif is_label and array.sum() == 0:
        # return [int(i) for i in np.random.choice(z, 1).tolist()] # a random slice could be all 0 as it was padded with 0.
        step = math.ceil(z/8) # will result in 7 or 8 points
        out = list(range(0, z, step))
    else:
        maxis = np.max(array, axis=(1,2)) > 0
        nonzero_indices = [i for i in np.nonzero(maxis)[0]]
        step = math.ceil(len(nonzero_indices)/8) # will result in 7 or 8 points
        out = list(range(nonzero_indices[0], nonzero_indices[-1], step))

    return out

def cuda2np(image):
    # image: d,h,w
    if isinstance(image, torch.Tensor):
        if image.is_cuda:
            image = image.cpu().detach().numpy()
        else:
            image = image.numpy()
    elif not isinstance(image, np.ndarray):
        logger.error('image should be torch.Tensor or numpy.ndarray')
    return image


def imagelist2figure(image, selected_z_list, colorslist, is_label=False,fig_title=''):
    # image: list of h,w numpy array images, or a numpy array of size(d,h,w)
    y_num = 2
    x_num = int(len(selected_z_list)/y_num) + 1
    # x_num = 4
    # y_num = int(len(selected_z_list)/x_num) + 1

    image_out = plt.figure(figsize=(2.2*y_num, 2.5*x_num))
    for i in range(len(selected_z_list)):
        ax = plt.subplot(x_num, y_num, i+1)

        img_temp = image[selected_z_list[i]]
        img_temp = img_temp[:, ::-1] # should be matched args to .imshow() below. # required to get image plotted in same direction as itk-snap
        if i == 0:
            ax.set_title('{}'.format(fig_title))
        ax.set_xlim(0, img_temp.shape[0])
        ax.set_ylim(0, img_temp.shape[1])
        if is_label:
            mycmaps = colors.ListedColormap(colorslist)
            ax.imshow(img_temp, cmap=mycmaps, origin='lower', vmin=0, vmax=len(colorslist)-1)
        else:
            ax.imshow(img_temp, cmap='gray', origin='lower')
    
    return image_out


class MySummaryWriter(SummaryWriter):
    '''
    '''
    def __init__(self, log_dir="./"):
        super(MySummaryWriter, self).__init__(log_dir=log_dir)
        
    def chooseSlices(self, array, is_label=True):
        array = cuda2np(array)
        if len(array.shape) == 2: # 2D
            return None
        elif len(array.shape) == 3:
            return selectSlices(array, is_label)

    def tensor2figure(self, image, selected_z_list, colorslist=['#000000','#00FF00','#0000FF','#FF0000', '#FFFF00'], is_label=False, fig_title=''):
        '''
        image: d, h, w
        colorslist: hex colors.
        
        steps: see train.tb_images().
        step1: selected_z_list = self.chooseSlices()
        step2: fig = self.tensor2figure(...,selected_z_list,...)
        step3: self.add_figure(fig)
        '''
        image = cuda2np(image)
        try:
            fig = imagelist2figure(image, selected_z_list, colorslist=colorslist, is_label=is_label, fig_title=fig_title)
        except Exception as e:
            logger.info('{}'.format(str(e)))
            # logger.info('image.shape:{}, selected_z_list:{}'.format(str(image.shape), str(selected_z_list)))

        return fig

    
