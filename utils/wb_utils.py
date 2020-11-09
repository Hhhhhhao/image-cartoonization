import numpy as np
import torch
import torch.nn.functional as F
from skimage import segmentation, color
from joblib import Parallel, delayed


def box_filter(x, r):
    ch = list(x.size())[1]
    
    weight = 1 / ((2*r+1) ** 2)

    box_kernel = weight * np.ones((ch, 1, 2*r+1, 2*r+1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    box_kernel = torch.from_numpy(box_kernel).to(x.device)
    output = F.conv2d(x, box_kernel, bias=None, stride=1, padding=(2*r+1)//2, groups=ch)
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
    batch_image = (batch_image + 1) / 2
    batch_image = batch_image * 255
    batch_image = batch_image.astype(np.uint8)

    def process_slic(image):
        seg_label = segmentation.slic(np.array(image), n_segments=seg_num, sigma=1, compactness=10, convert2lab=True)
        image = color.label2rgb(seg_label, np.array(image), kind='avg')
        return image

    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)\
                         (image) for image in batch_image)

    batch_out = np.asarray(batch_out)
    batch_out = batch_out / 255.
    batch_out = batch_out * 2 - 1
    batch_out = batch_out.astype(np.float32)
    return batch_out.transpose(0, 3, 1, 2)