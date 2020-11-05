import os
import logging
from datetime import datetime
from logging import Formatter
from logging.handlers import RotatingFileHandler
from .misc import ensure_dir, read_json, write_json


def setup_logging(log_dir):
    log_file_format = '[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d'
    log_console_format = '[%(levelname)s]: %(message)s'

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}/exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=20)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}/exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def print_configs(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)


def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name + '_{}'.format(config.data_aug_policy.replace(',', '_')) + '_{}_{}_{}_bs{}_glr{}_dlr{}_wd{}'.format(config.src_style, config.tar_style, config.image_size, config.batch_size, config.g_lr, config.d_lr, config.weight_decay)
    exp_name += ('_' + timestamp)

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join('experiments', exp_name, 'tb')
    config.checkpoint_dir = os.path.join('experiments', exp_name, 'checkpoints/')
    config.log_dir = os.path.join('experiments', exp_name, 'logs/')
    config.result_dir = os.path.join('experiments', exp_name, 'results/')
    for dir in [config.summary_dir, config.checkpoint_dir, config.log_dir, config.result_dir]:
        ensure_dir(dir)

    # setup logging in the project
    setup_logging(config.log_dir)
    logging.getLogger().info('The pipeline of the project will begin now.')
    print_configs(config)
    # save config
    write_json(vars(config), os.path.join('experiments', exp_name, 'config.json'))

    return config