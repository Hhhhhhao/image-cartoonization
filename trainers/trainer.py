import torch
from torchvision.utils import make_grid
import numpy as np
from base import BaseTrainer
from utils import MetricTracker
from utils.metric import accuracy
from utils.misc import Normalize

class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)

        # setup visualization writer instance
        self.logger.info("Creating tensorboard writer...")
        self.writer = TensorboardWriter(config.summary_dir, self.logger, config.tensorboard)

        self.logger.info("Creating data loaders...")
        self.train_dataloader, self.valid_dataloader = self._build_dataloader()
        self.log_step = int(np.sqrt(self.train_dataloader.batch_size))

        self.logger.info("Creating model architecture...")
        model = self._build_model()
        self.model = model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        self.logger.info("Creating optimizers...")
        self.optimizer, self.lr_scheduler = self._build_optimizer(self.model)

        if self.config.resume is not None:
            self._resume_checkpoint(config.resume)

        # build loss
        self.logger.info("Creating losses...")
        self._build_criterion()

        self.logger.info("Creating metric trackers...")
        self._build_metrics()

    def _build_model(self):
        """ build generator and discriminator model """
        model = None
        return model

    def _build_criterion(self):
        criterion = None
        return criterion

    def _build_metrics(self):
        self.metric_names = [
                   'cls_loss', 'cls_loss_gender', 'cls_loss_emotion', 'cls_loss_pose',
                   'cls_acc', 'cls_acc_gender', 'cls_acc_emotion', 'cls_acc_pose']
        self.train_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)
        self.valid_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
