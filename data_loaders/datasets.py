import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.cartoongan import smooth_image_edges


class CartoonDataset(Dataset):
    def __init__(self, data_dir, src_style='real', tar_style='gongqijun', src_transform=None, tar_transform=None):
        self.data_dir = data_dir
        self.src_data, self.tar_data = self._load_data(data_dir, src_style, tar_style)
        print("total {} {} images for training".format(len(self.src_data), src_style))
        print("total {} {} images for training".format(len(self.tar_data), tar_style))
        self.src_transform = src_transform
        self.tar_transform = tar_transform

    def _load_data(self, data_dir, src_style, tar_style):
        src_data = []
        with open(os.path.join(data_dir, '{}_train.txt'.format(src_style)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                path = line.strip()
                src_data.append(path)

        tar_data = []
        with open(os.path.join(data_dir, '{}_train.txt'.format(tar_style)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                path = line.strip()
                tar_data.append(path)

        return src_data, tar_data

    def _shuffle_data(self):
        np.random.shuffle(self.src_data)
        np.random.shuffle(self.tar_data)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, index):
        src_path = self.src_data[index]
        tar_path = self.tar_data[index]
        src_img = Image.open(os.path.join(self.data_dir, src_path))
        tar_img = Image.open(os.path.join(self.data_dir, tar_path))
        src_img = src_img.convert('RGB')
        tar_img = tar_img.convert('RGB')

        # transform src img
        if self.src_transform is not None:
            src_img = self.src_transform(src_img)
        # transform tar img
        if self.tar_transform is not None:
            tar_img = self.tar_transform(tar_img)
        return src_img, tar_img


class CartoonDefaultDataset(Dataset):
    def __init__(self, data_dir, style='real', transform=None):
        self.data_dir = data_dir
        self.data = self._load_data(data_dir, style)
        print("total {} {} images for testing".format(len(self.data), style))
        self.transform = transform

    def _load_data(self, data_dir, style):
        data = []
        with open(os.path.join(data_dir, '{}_test.txt'.format(style)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                path = line.strip()
                data.append(path)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')

        # transform src img
        if self.transform is not None:
            img = self.transform(img)
        return img


class CartoonGANDataset(CartoonDataset):
    def __init__(self, data_dir, src_style='real', tar_style='gongqijun', src_transform=None, tar_transform=None):
        super(CartoonGANDataset, self).__init__(data_dir, src_style, tar_style, src_transform, tar_transform)

    def __getitem__(self, index):
        src_path = self.src_data[index]
        tar_path = self.tar_data[index]
        src_img = Image.open(os.path.join(self.data_dir, src_path))
        tar_img = Image.open(os.path.join(self.data_dir, tar_path))
        src_img = src_img.convert('RGB')
        tar_img = tar_img.convert('RGB')

        # get edge smoothed transform
        smooth_tar_img = smooth_image_edges(np.asarray(tar_img))
        smooth_tar_img = Image.fromarray(smooth_tar_img)

        # transform src img
        if self.src_transform is not None:
            src_img = self.src_transform(src_img)
        # transform tar img
        if self.tar_transform is not None:
            tar_img = self.tar_transform(tar_img)
            smooth_tar_img = self.tar_transform(smooth_tar_img)
        return src_img, tar_img, smooth_tar_img


class StarCartoonDataset(Dataset):
    def __init__(self, data_dir, real_transform=None, car_transform=None):
        self.data_dir = data_dir
        self.data = self._load_data(data_dir)
        self.real_transform = real_transform
        self.car_transform = car_transform

    def _load_data(self, data_dir):
        styles = ['real', 'gongqijun', 'xinhaicheng']
        data = {}
        for i, style in enumerate(styles):
            data[i] = []
            with open(os.path.join(data_dir, '{}_train.txt'.format(style)), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    path = line.strip()
                    data[i].append(path)
        return data

    def _shuffle_data(self):
        for key, item in self.data.items():
            np.random.shuffle(item)
            self.data[key] = item

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        # sample a source
        src_label = random.randint(0, 2)
        # sample a target
        tar_label = random.randint(0, 2)
        src_path = self.data[src_label][index]
        tar_path = self.data[tar_label][index]
        src_img = Image.open(os.path.join(self.data_dir, src_path))
        tar_img = Image.open(os.path.join(self.data_dir, tar_path))
        src_img = src_img.convert('RGB')
        tar_img = tar_img.convert('RGB')

        if src_label == 0:
            src_img = self.real_transform(src_img)
        else:
            src_img = self.car_transform(src_img)

        if tar_label == 0:
            tar_img = self.real_transform(tar_img)
        else:
            tar_img = self.car_transform(tar_img)

        return src_img, src_label, tar_img, tar_label


if __name__ == '__main__':
    from tqdm import tqdm
    data_dir = '/home/zhaobin/cartoon/'
    style = 'gongqijun'
    dataset = StarCartoonDataset(data_dir)
    import matplotlib.pyplot as plt

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        src_img, tar_img, tar_label = dataset.__getitem__(i)


