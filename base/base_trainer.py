import torch
import numpy as np
import logging
from abc import abstractmethod
from numpy import inf
from utils import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,  config):
        self.config = config
        self.logger = logging.getLogger("Training")
        self.epochs = config.epochs
        self.save_period = config.save_period
        self.start_epoch = 1
        self.config.train = True

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config.n_gpu)

        # setup visualization writer instance
        self.logger.info("Creating tensorboard writer...")
        self.writer = TensorboardWriter(config.summary_dir, self.logger, config.tensorboard)

    def _build_model(self):
        """ build model """
        raise NotImplementedError

    def _build_optimizer(self, *kwargs):
        """ build optimizer """
        raise NotImplementedError

    def _build_dataloader(self):
        """ build train and validation data loader """
        raise NotImplementedError

    @abstractmethod
    def _build_criterion(self):
        """ build loss functions """
        raise NotImplementedError

    @abstractmethod
    def _build_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # save checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        self.logger.info("useing device {}".format(device))
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model = type(self.model).__name__
        model = model[:-9]
        state = {
            "model": model,
            'epoch': epoch,
            'state_dict': self.model.state_dict() if len(self.device_ids) <= 1 else self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = str(self.config.checkpoint_dir + 'current.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if epoch % self.save_period == 0:
            filename = str(self.config.checkpoint_dir + 'epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['model'] != self.config.model:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.train_dataloader.batch_size
        total = self.train_dataloader.n_samples
        return base.format(current, total, 100.0 * current / total)
