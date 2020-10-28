from torchvision import transforms
from base import BaseDataLoader
from .datasets import CartoonDataset


def build_transform(style='real', image_size=256):
    if style == 'real':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    else:

        transform = transforms.Compose([
            transforms.RandomCrop(512),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transform


class CartoonDataLoader(BaseDataLoader):
    def __init__(self, data_dir, src_style='real', tar_style='gongqijun', image_size=256, batch_size=16, num_workers=4, validation_split=0.01):

        # data augmentation
        src_transform = build_transform(src_style, image_size)
        tar_transform = build_transform(tar_style, image_size)

        # create dataset
        self.dataset = CartoonDataset(data_dir, src_style, tar_style, src_transform, tar_transform)

        super(CartoonDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            num_workers=num_workers,
            drop_last=True)

    def shuffle_dataset(self):
        self.dataset._shuffle_data()


if __name__ == '__main__':
    data_dir = '/Users/leon/Downloads/cartoon_datasets'
    style = 'gongqijun'
    data_loader = CartoonDataLoader(data_dir)
    valid_dataloader = data_loader.split_validation()

    for i, (src_img, tar_img) in enumerate(data_loader):
        print(src_img.shape)
        print(tar_img.shape)

