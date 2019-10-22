#### @Chao Huang(huangchao09@zju.edu.cn).
import numpy as np

import config

def num_pool2stride_size(num_pool_per_axis):
    max_num = max(num_pool_per_axis)
    stride_size_per_pool = list()
    for i in range(max_num):
        unit = [1,2]
        stride_size_per_pool.append([unit[i<num_pool_per_axis[0]], unit[i<num_pool_per_axis[1]], unit[i<num_pool_per_axis[2]]])
    return stride_size_per_pool
    

