import numpy as np
import torch
import torch.nn.functional as F
from skimage import segmentation, color
from joblib import Parallel, delayed
from skimage.color import rgb2hsv, rgb2lab, rgb2grey
from skimage.segmentation import find_boundaries
from scipy.ndimage import find_objects
from skimage.feature import local_binary_pattern


def _calculate_color_sim(ri, rj):
    """
        Calculate color similarity using histogram intersection
    """
    return sum([min(a, b) for a, b in zip(ri["color_hist"], rj["color_hist"])])


def _calculate_texture_sim(ri, rj):
    """
        Calculate texture similarity using histogram intersection
    """
    return sum([min(a, b) for a, b in zip(ri["texture_hist"], rj["texture_hist"])])


def _calculate_size_sim(ri, rj, imsize):
    """
        Size similarity boosts joint between small regions, which prevents
        a single region from engulfing other blobs one by one.
        size (ri, rj) = 1 − [size(ri) + size(rj)] / size(image)
    """
    return 1.0 - (ri['size'] + rj['size']) / imsize


def _calculate_fill_sim(ri, rj, imsize):
    """
        Fill similarity measures how well ri and rj fit into each other.
        BBij is the bounding box around ri and rj.
        fill(ri, rj) = 1 − [size(BBij) − size(ri) − size(ri)] / size(image)
    """

    bbsize = (max(ri['box'][2], rj['box'][2]) - min(ri['box'][0], rj['box'][0])) * (max(ri['box'][3], rj['box'][3]) - min(ri['box'][1], rj['box'][1]))

    return 1.0 - (bbsize - ri['size'] - rj['size']) / imsize


def calculate_color_hist(mask, img):
    """
        Calculate colour histogram for the region.
        The output will be an array with n_BINS * n_color_channels.
        The number of channel is varied because of different
        colour spaces.
    """

    BINS = 25
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    channel_nums = img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)

    return hist


def generate_lbp_image(img):

    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    channel_nums = img.shape[2]

    lbp_img = np.zeros(img.shape)
    for channel in range(channel_nums):
        layer = img[:, :, channel]
        lbp_img[:, :,channel] = local_binary_pattern(layer, 8, 1)

    return lbp_img


def calculate_texture_hist(mask, lbp_img):
    """
        Use LBP for now, enlightened by AlpacaDB's implementation.
        Plan to switch to Gaussian derivatives as the paper in future
        version.
    """

    BINS = 10
    channel_nums = lbp_img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = lbp_img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)

    return hist


def calculate_sim(ri, rj, imsize, sim_strategy):
    """
        Calculate similarity between region ri and rj using diverse
        combinations of similarity measures.
        C: color, T: texture, S: size, F: fill.
    """
    sim = 0

    if 'C' in sim_strategy:
        sim += _calculate_color_sim(ri, rj)
    if 'T' in sim_strategy:
        sim += _calculate_texture_sim(ri, rj)
    if 'S' in sim_strategy:
        sim += _calculate_size_sim(ri, rj, imsize)
    if 'F' in sim_strategy:
        sim += _calculate_fill_sim(ri, rj, imsize)

    return sim

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


def selective_adacolor(batch_image, seg_num=200, power=1):
    return color_ss_map(image=batch_image, seg_num=seg_num, power=power)


def color_ss_map(image, seg_num=100, power=1,
                 color_space='Lab', k=10, sim_strategy='CTSF'):
    image = (image.astype(np.float32) / 255.) * 2 - 1
    img_seg = segmentation.felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
    img_cvtcolor = label2rgb(img_seg, image, kind='mix')
    img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
    S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    # Start hierarchical grouping

    while S.num_regions() > seg_num:
        i, j = S.get_highest_similarity()
        S.merge_region(i, j)
        S.remove_similarities(i, j)
        S.calculate_similarity_for_new_region()

    image = label2rgb(S.img_seg, image, kind='mix')
    image = (image + 1) / 2
    image = image ** power
    image = image / np.max(image)
    image *= 255
    return image.astype(np.uint8)


def label2rgb(label_field, image, kind='mix', bg_label=-1, bg_color=(0, 0, 0)):

    #std_list = list()
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        #std = np.std(image[mask])
        #std_list.append(std)
        if kind == 'avg':
            color = image[mask].mean(axis=0)
        elif kind == 'median':
            color = np.median(image[mask], axis=0)
        elif kind == 'mix':
            std = np.std(image[mask])
            if std < 20:
                color = image[mask].mean(axis=0)
            elif 20 < std < 40:
                mean = image[mask].mean(axis=0)
                median = np.median(image[mask], axis=0)
                color = 0.5*mean + 0.5*median
            elif 40 < std:
                color = np.median(image[mask], axis=0)
        out[mask] = color
    return out


def switch_color_space(img, target):
    """
        RGB to target color space conversion.
        I: the intensity (grey scale), Lab, rgI: the rg channels of
        normalized RGB plus intensity, HSV, H: the Hue channel H from HSV
    """

    if target == 'HSV':
        return rgb2hsv(img)

    elif target == 'Lab':
        return rgb2lab(img)

    elif target == 'I':
        return rgb2grey(img)

    elif target == 'rgb':
        img = img / np.sum(img, axis=0)
        return img

    elif target == 'rgI':
        img = img / np.sum(img, axis=0)
        img[:,:,2] = rgb2grey(img)
        return img

    elif target == 'H':
        return rgb2hsv(img)[:,:,0]

    else:
        raise "{} is not suported.".format(target)


class HierarchicalGrouping(object):
    def __init__(self, img, img_seg, sim_strategy):
        self.img = img
        self.sim_strategy = sim_strategy
        self.img_seg = img_seg.copy()
        self.labels = np.unique(self.img_seg).tolist()

    def build_regions(self):
        self.regions = {}
        lbp_img = generate_lbp_image(self.img)
        for label in self.labels:
            size = (self.img_seg == 1).sum()
            region_slice = find_objects(self.img_seg == label)[0]
            box = tuple([region_slice[i].start for i in (1, 0)] +
                        [region_slice[i].stop for i in (1, 0)])

            mask = self.img_seg == label
            color_hist = calculate_color_hist(mask, self.img)
            texture_hist = calculate_texture_hist(mask, lbp_img)

            self.regions[label] = {
                'size': size,
                'box': box,
                'color_hist': color_hist,
                'texture_hist': texture_hist
            }

    def build_region_pairs(self):
        self.s = {}
        for i in self.labels:
            neighbors = self._find_neighbors(i)
            for j in neighbors:
                if i < j:
                    self.s[(i, j)] = calculate_sim(self.regions[i],
                                                           self.regions[j],
                                                           self.img.size,
                                                           self.sim_strategy)

    def _find_neighbors(self, label):
        """
            Parameters
        ----------
            label : int
                label of the region
        Returns
        -------
            neighbors : list
                list of labels of neighbors
        """

        boundary = find_boundaries(self.img_seg == label,
                                   mode='outer')
        neighbors = np.unique(self.img_seg[boundary]).tolist()

        return neighbors

    def get_highest_similarity(self):
        return sorted(self.s.items(), key=lambda i: i[1])[-1][0]

    def merge_region(self, i, j):

        # generate a unique label and put in the label list
        new_label = max(self.labels) + 1
        self.labels.append(new_label)

        # merge blobs and update blob set
        ri, rj = self.regions[i], self.regions[j]

        new_size = ri['size'] + rj['size']
        new_box = (min(ri['box'][0], rj['box'][0]),
                   min(ri['box'][1], rj['box'][1]),
                   max(ri['box'][2], rj['box'][2]),
                   max(ri['box'][3], rj['box'][3]))
        value = {
            'box': new_box,
            'size': new_size,
            'color_hist':
                (ri['color_hist'] * ri['size']
                 + rj['color_hist'] * rj['size']) / new_size,
            'texture_hist':
                (ri['texture_hist'] * ri['size']
                 + rj['texture_hist'] * rj['size']) / new_size,
        }

        self.regions[new_label] = value

        # update segmentation mask
        self.img_seg[self.img_seg == i] = new_label
        self.img_seg[self.img_seg == j] = new_label

    def remove_similarities(self, i, j):

        # mark keys for region pairs to be removed
        key_to_delete = []
        for key in self.s.keys():
            if (i in key) or (j in key):
                key_to_delete.append(key)

        for key in key_to_delete:
            del self.s[key]

        # remove old labels in label list
        self.labels.remove(i)
        self.labels.remove(j)

    def calculate_similarity_for_new_region(self):
        i = max(self.labels)
        neighbors = self._find_neighbors(i)

        for j in neighbors:
            # i is larger than j, so use (j,i) instead
            self.s[(j, i)] = calculate_sim(self.regions[i],
                                                   self.regions[j],
                                                   self.img.size,
                                                   self.sim_strategy)

    def is_empty(self):
        return True if not self.s.keys() else False

    def num_regions(self):
        return len(self.s.keys())