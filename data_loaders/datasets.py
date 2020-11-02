import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


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


if __name__ == '__main__':
    from tqdm import tqdm
    data_dir = '/Users/leon/Downloads/cartoon_datasets'
    style = 'gongqijun'
    dataset = CartoonDataset(data_dir, style)

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        src_img, tar_img = dataset.__getitem__(i)
