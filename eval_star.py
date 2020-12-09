import os
working_dir = os.path.dirname(__file__)
import argparse
import torch
from easydict import EasyDict as edict
from tqdm import tqdm
from data_loaders import CartoonDefaultDataLoader
from models import StarGenerator, MappingNetwork
from utils.misc import read_json
import numpy as np
import cv2
from fid_score import calculate_fid_given_paths
from kid_score import calculate_kid_given_paths
from acc_score import compute_acc_score
from utils.wb_utils import guided_filter

def get_config(manual=None):
    parser = argparse.ArgumentParser('Image Cartoon')
    # basic options
    parser.add_argument('--checkpoint-path', default='experiments/cyclegan_color_translation_cutout_real_gongqijun_128_bs12_glr0.0001_dlr0.0002_wd0.0001_201106_025817/checkpoints/current.pth', help='checkpoint path')
    parser.add_argument('--image-size', default=128, type=int, help='image size')
    return parser.parse_args(manual)


def main():
    config = get_config()
    image_size = config.image_size

    # find config.json in checkpoint folder
    checkpoint_path = os.path.join(working_dir, config.checkpoint_path)
    checkpoint_epoch = checkpoint_path.split('/')[-1].split('.')[0]
    checkpoint_dir = os.path.dirname(checkpoint_path)
    exp_dir = os.path.dirname(checkpoint_dir)
    result_dir = os.path.join(exp_dir, 'results')

    # make device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load config
    config = read_json(os.path.join(exp_dir, 'config.json'))
    config = edict(config)

    # build dataloader
    data_loader = CartoonDefaultDataLoader(
        data_dir=config.data_dir,
        style=config.src_style,
        image_size=image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers)

    # build model
    model = StarGenerator(config.image_size, config.down_size, config.num_res, config.skip_conn)
    mapnet  = MappingNetwork(latent_dim=16, style_dim=config.style_size, num_domains=4)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['gen_state_dict'])
    mapnet.load_state_dict(checkpoint['map_state_dict'])
    model.to(device)
    mapnet.to(device)
    model.eval()
    mapnet.eval()

    # start evaluation
    print("start evaluation")
    for i, tar_style in enumerate(['gongqijun', 'xinhaicheng', 'disney', 'tangqian']):
        image_dir = os.path.join(result_dir, '{}2{}_{}_{}'.format(config.src_style, tar_style, image_size, checkpoint_epoch))
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        count = 0
        with torch.no_grad():
            for batch_idx, src_imgs in tqdm(enumerate(data_loader), total=len(data_loader)):
                src_imgs = src_imgs.to(device)
                tar_labels = torch.ones((src_imgs.size(0)), dtype=torch.long) * i

                tar_z = torch.randn((src_imgs.size(0), config.latent_size)).to(device)
                tar_s = mapnet(tar_z, tar_labels)
                tar_imgs = model(src_imgs, tar_s)

                # save images
                tar_imgs = tar_imgs.cpu().numpy().transpose(0, 2, 3, 1)
                # convert from [-1, 1] to [0, 255] uint
                tar_imgs = (tar_imgs + 1) / 2
                tar_imgs = (tar_imgs * 255).astype(np.uint8)

                src_imgs = src_imgs.cpu().numpy().transpose(0, 2, 3, 1)
                # convert from [-1, 1] to [0, 255] uint
                src_imgs = (src_imgs + 1) / 2
                src_imgs = (src_imgs * 255).astype(np.uint8)

                for src_img, tar_img in zip(src_imgs, tar_imgs):
                    cv2.imwrite(os.path.join(image_dir, '{}_tar.png'.format(count)), tar_img[:, :, ::-1])
                    cv2.imwrite(os.path.join(image_dir, '{}_src.png'.format(count)), src_img[:, :, ::-1])
                    count += 1

        result_file = open('{}/{}2_{}_result_{}_{}.txt'.format(result_dir, config.src_style, tar_style, image_size, checkpoint_epoch), "w")

        result_file.write("Style {}\n".format(tar_style))

        # calculate acc score
        results = compute_acc_score(
            '/'.join(image_dir.split('/')[:-1]),
            '/home/ubuntu/haoc/image-cartoonization/experiments/classifier_color_translation_cutout_real_gongqijun_128_bs128_glr0.001_dlr0.0002_wd0.0001_201209_034822/checkpoints/current.pth',
            tar_style,
            image_size)
        line = 'Acc: %.4f (%.4f) \n' % (results[0], results[1])
        result_file.write(line)
        print(line)

        # calculate fid score
        results = calculate_fid_given_paths(['{}/{}_test.txt'.format(config.data_dir, tar_style), image_dir], config.batch_size, torch.cuda.is_available(), 2048, model_type='inception')
        for p, m, s in results:
            line = 'FID (%s): %.4f (%.4f)\n' % (p, m, s)
            result_file.write(line)
            print(line)

        # calculate kid score
        results = calculate_kid_given_paths(['{}/{}_test.txt'.format(config.data_dir, tar_style), image_dir], config.batch_size, torch.cuda.is_available(), 2048, model_type='inception')
        for p, m, s in results:
            line = 'KID (%s): %.4f (%.4f)\n' % (p, m, s)
            result_file.write(line)
            print(line)


if __name__ == '__main__':
    main()