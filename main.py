import argparse
from utils import process_config
from trainers import Trainer


def get_config(manual=None):

    parser = argparse.ArgumentParser('Experiment Name')

    # basic options
    parser.add_argument('--exp-name', default='template', help='experiment name')
    parser.add_argument('--data-dir', default='None', help='data dir')
    parser.add_argument('--n-gpu', default=1, type=int, help='number of gpus to use')
    parser.add_argument('--tensorboard', default=False, action='store_true', help='use tensorboard to log results')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers in data loaders')
    parser.add_argument('--save-period', default=1, type=int, help='saving period for models')
    parser.add_argument('--resume', default=None, help='resume checkpoint path')

    # train options
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs for training the model')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='generator learning rate')
    parser.add_argument('--weight-decay', default=1e-3, type=float, help='weight decay for optimizers')
    parser.add_argument('--monitor', default='max val_cls_acc', help='monitor metric for early stopping')
    parser.add_argument('--early-stop', type=int, default=5, help='early stopping metric')

    # model options

    return parser.parse_args(manual)


def main():
    config = get_config()
    config = process_config(config)

    # trainer
    trainer = build_trainer(config)

    # train
    trainer.train()


if __name__ == '__main__':
    main()
