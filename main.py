import argparse
from utils import process_config
from trainers import build_trainer


def init_config():
    parser = argparse.ArgumentParser('Image Cartoon')

    # basic options
    parser.add_argument('--exp-name', default='stargan', help='experiment name',
                        choices=['cyclegan', 'cartoongan', 'whitebox', 'stargan', 'classifier'])
    parser.add_argument('--data-dir', default='/home/zhaobin/cartoon/', help='data dir')
    parser.add_argument('--n-gpu', default=1, type=int, help='number of gpus to use')
    parser.add_argument('--tensorboard', default=False, action='store_true', help='use tensorboard to log results')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers in data loaders')
    parser.add_argument('--save-period', default=10, type=int, help='saving period for models')
    parser.add_argument('--resume', default=None, help='resume checkpoint path')

    # train options
    parser.add_argument('--src-style', default='real', help='source images style', choices=['real'])
    parser.add_argument('--tar-style', default='gongqijun', help='target images style', choices=['gongqijun', 'tangqian', 'xinhaicheng', 'disney'])
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs for training the model')
    parser.add_argument('--batch-size', default=2, type=int, help='batch size')
    parser.add_argument('--g-lr', default=1e-4, type=float, help='generator learning rate')
    parser.add_argument('--d-lr', default=2e-4, type=float, help='discriminator learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay for optimizers')
    parser.add_argument('--monitor', default=None, help='monitor metric for early stopping')
    parser.add_argument('--early-stop', type=int, default=5, help='early stopping metric')
    parser.add_argument('--adv-criterion', type=str, default='LSGAN', help='adversarial loss type')
    parser.add_argument('--lambda-adv', type=float, default=1, help='adversarial loss weight')

    # model options
    parser.add_argument('--image-size', default=128, type=int, help='input size')
    parser.add_argument('--down-size', default=16, type=int, help='downsample size')
    parser.add_argument('--num-res', default=4, type=int, help='number of residual blocks in image generator')
    parser.add_argument('--skip-conn', default=False, action='store_true', help="flag of using skip connection in generator")
    parser.add_argument('--data-aug-policy', default='color,translation,cutout', help='data efficient gan training')

    # ================== extra options: add parameters in your experiements here =========================
    # cyclegan
    parser.add_argument('--lambda-rec', type=float, default=10, help='cycle rec. loss weight')

    # whiteboxgan
    parser.add_argument('--lambda-tv', type=float, default=1, help='total variance loss')

    # star
    parser.add_argument('--style-size', type=int, default=64, help='style dimension')
    parser.add_argument('--latent-size', type=int, default=16, help='latent code dimension')
    parser.add_argument('--lambda-sty', type=float, default=1, help='style loss weight')
    parser.add_argument('--lambda-ds', type=float, default=1, help='diversity loss weight')
    parser.add_argument('--lambda-cls', type=float, default=1, help='classification loss weight')
    parser.add_argument('--nce-t', type=float, default=0.07)

    # classifier
    parser.add_argument('--num-feature', type=int, default=1024, help='num of features')
    parser.add_argument('--num-class', type=int, default=4, help='num of classes')

    return parser.parse_args()


def override_config(config):
    # set down-size according to image size. DO Not change this!
    if config.image_size == 128:
        config.down_size = 16
        config.num_res = 4
    elif config.image_size == 256:
        config.down_size = 32
        config.num_res = 8

    return config


def main():
    # get basic config
    config = init_config()
    # override options
    config = override_config(config)
    # process config
    config = process_config(config)

    # build trainer
    trainer = build_trainer(config)

    # train
    trainer.train()


if __name__ == '__main__':
    main()
