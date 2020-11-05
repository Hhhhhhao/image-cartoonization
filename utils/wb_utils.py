import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
from skimage import segmentation, color


def box_filter(x, r):
    ch = list(x.size())[1]
    
    weight = 1 / ((2*r+1) ** 2)

    box_kernel = weight * np.ones((1, ch, 2*r+1, 2*r+1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    box_kernel = torch.from_numpy(box_kernel).to(x.device)
    output = F.conv2d(x, box_kernel, bias=None, stride=1, padding=(2*r+1)//2)
    return output


def guided_filter(x, y, r, eps=1e-2):
    x_shape = list(x.size())

    N = box_filter(torch.ones((1, 1, x_shape[2], x_shape[3])), r).to(x.device)

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
    r1, g1, b1 = image1[:, 0, :, :], image1[:, 1, :, :], image1[:, 2, :, :]
    if mode == 'normal':
        r_weight = torch.normal(mean=0.299, std=0.1)
        g_weight = torch.normal(mean=0.587, std=0.1)
        b_weight = torch.normal(mean=0.114, std=0.1)
    elif mode == 'uniform':
        r_weight = (0.399-0.199) * torch.rand(1) + 0.199
        g_weight = (0.687-0.487) * torch.rand(1) + 0.487
        b_weight = (0.214-0.014) * torch.rand(1) + 0.014
    r_weight = r_weight.to(image1.device)
    g_weight = g_weight.to(image1.device)
    b_weight = b_weight.to(image1.device)
    output1 = (r_weight*r1 + g_weight*g1 + b_weight*b1) / (r_weight+g_weight+b_weight)  
    return output1.unsqueeze(1).repeat(1, 3, 1, 1)


def superpixel(batch_image, seg_num=100):
    def process_slic(image):
        seg_label = segmentation.slic(np.array(image), n_segments=seg_num, sigma=1,
                                        compactness=10, convert2lab=True)
        image = color.label2rgb(seg_label, np.array(image), kind='avg')
        return image
    
    # num_job = np.shape(batch_image)[0]
    batch_out = process_slic(batch_image)
    return np.array(batch_out)


def imageblur(kernel_size, image, device):
    channel = list(image.size())[1]
    return GaussianBlur(channel, 2*kernel_size+1)(image).to(device)


class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size):
        super(GaussianBlur, self).__init__()
        # gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        # gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        x = np.linspace(-1, 1, kernel_size+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        kern2d = kern2d/kern2d.sum()
        kern2d = torch.Tensor(kern2d)

        kern2d = kern2d.view(1, 1, kernel_size, kernel_size)
        kern2d = kern2d.repeat(channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, bias=False)

        self.gaussian_filter.weight.data = kern2d

    def forward(self, x):
        return self.gaussian_filter(x)


