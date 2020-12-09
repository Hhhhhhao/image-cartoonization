import os
import numpy as np 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.datasets import ImageFolder
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models import ResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AvgMeter():
    def __init__(self):
        self.qty = 0
        self.cnt = 0

    def update(self, increment, count):
        self.qty += increment
        self.cnt += count

    def get_avg(self):
        if self.cnt == 0:
            return 0
        else:
            return self.qty/self.cnt


def compute_acc_score(image_path, model_path, image_style, image_size=256, num_feature=1024, num_class=4):

    transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    image_data = ImageFolder(image_path, transform=transform)
    image_loader = DataLoader(image_data, batch_size=32, drop_last=False, pin_memory=True, shuffle=False)

    model = ResNet(num_feature, num_class)
    model.load_state_dict(torch.load(model_path)['resnet_state_dict'])
    model.to(device)
    model.eval()

    acc_meter = AvgMeter()

    class_dict = {
        "disney": 0,
        "gongqijun": 1,
        "tangqian": 2,
        "xinhaicheng": 3,
    }
    gt = class_dict[image_style]
    print("gt", gt)

    with torch.no_grad():
        for i, (data, _) in enumerate(image_loader):
            data = data.to(device)
            pred = model(data)

            label = pred.argmax(1).eq(gt)
            label = label.view(-1).float()

            acc_meter.update(np.sum(np.equal(label.cpu().numpy(), gt)), data.size(0))

    return acc_meter.get_avg()


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--image-path', type=str, required=True,
                        help='path to image folder')
    parser.add_argument('--model-path', type=str, default='experiments',
                        help='path to model')
    parser.add_argument('--image-size', type=int, default=224,
                        help='resized image size')
    parser.add_argument('--image-style', type=str, required=True,
                        help='verification image style')
    parser.add_argument('--num-feature', type=int, default=1024,
                        help='num of features for resnet linear')
    parser.add_argument('--num-class', type=int, default=4,
                        help='num of classes for resnet output')
    parser.add_argument('--verif-model', type=str, required=True,
                        help='model name where generate cartoon images')

    args = parser.parse_args()

    acc = compute_acc_score(args.image_path, args.model_path, args.image_style, args.image_size, args.num_feature, args.num_class)
    print("The Acc Score of {} in translation to {} style is: {:4f}%".format(args.verif_model, args.image_style, acc))


# python acc_score.py --image-path ./test --model-path ./current.pth --image-style disney --verif-model default

    