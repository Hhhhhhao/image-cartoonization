import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import segmentation, color

def box_filter(x, r):
    ch = list(x.size())[1]
    
    weight = 1 / ((2*r+1) ** 2)

    box_kernel = weight * np.ones((1, ch, 2*r+1, 2*r+1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    box_kernel = torch.from_numpy(box_kernel)

    output = F.conv2d(x, box_kernel, bias=None, stride=1, padding=r, groups=3)
    return output

def guided_filter(x, y, r, eps=1e-2):
    x_shape = list(x.size())

    N = box_filter(torch.ones((1, 1, x_shape[1], x_shape[2])), r)

    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x*y, r) / N - mean_x * mean_y
    var_x = box_filter(x*x, r) /N - mean_x * mean_y

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output

def color_shift_2image(image1, image2, mode='uniform'):
    r1, g1, b1 = image1[0], image1[1], image1[2]
    r2, g2, b2 = image2[0], image2[1], image2[2]
    if mode == 'normal':
        r_weight = torch.normal(mean=0.299, std=0.1)
        g_weight = torch.normal(mean=0.587, std=0.1)
        b_weight = torch.normal(mean=0.114, std=0.1)

    elif mode == 'uniform':
        r_weight = (0.399-0.199) * torch.rand(1) + 0.199
        g_weight = (0.687-0.487) * torch.rand(1) + 0.487
        b_weight = (0.214-0.014) * torch.rand(1) + 0.014
    
    output1 = (r_weight*r1 + g_weight*g1 + b_weight*b1) / (r_weight+g_weight+b_weight)  
    output2 = (r_weight*r2 + g_weight*g2 + b_weight*b2) / (r_weight+g_weight+b_weight)  
    return output1, output2


def color_shift(image1, mode='uniform'):
    r1, g1, b1 = image1[0], image1[1], image1[2]
    if mode == 'normal':
        r_weight = torch.normal(mean=0.299, std=0.1)
        g_weight = torch.normal(mean=0.587, std=0.1)
        b_weight = torch.normal(mean=0.114, std=0.1)

    elif mode == 'uniform':
        r_weight = (0.399-0.199) * torch.rand(1) + 0.199
        g_weight = (0.687-0.487) * torch.rand(1) + 0.487
        b_weight = (0.214-0.014) * torch.rand(1) + 0.014
    
    output1 = (r_weight*r1 + g_weight*g1 + b_weight*b1) / (r_weight+g_weight+b_weight)  
    return output1

def superpixel(image, seg_num=200):
    seg_label = segmentation.slic(image.numpy(), n_segments=seg_num, sigma=1, compactness=10, convert2lab=True)
    image = color.label2rgb(seg_label, image.numpy(), kind='mix')
    return torch.from_numpy(image)
