import torch
from torchvision.utils import make_grid
import numpy as np
from base import BaseTrainer
from models import Generator, Discriminator
from losses import *
from data_loaders import CartoonDataLoader
from utils import MetricTracker


class ExpnameTrainer(BaseTrainer):
    def __init__(self, config):
        super(ExpnameTrainer, self).__init__(config)

        self.logger.info("Creating data loaders...")
        self.train_dataloader, self.valid_dataloader = self._build_dataloader()
        self.log_step = int(np.sqrt(self.train_dataloader.batch_size))

        self.logger.info("Creating model architecture...")
        gen, disc = self._build_model()
        # resume
        if self.config.resume is not None:
            self._resume_checkpoint(config.resume)
        # move to device
        self.gen = gen.to(self.device)
        self.disc = disc.to(self.device)
        if len(self.device_ids) > 1:
            self.gen = torch.nn.DataParallel(self.gen, device_ids=self.device_ids)
            self.disc = torch.nn.DataParallel(self.disc, device_ids=self.device_ids)

        self.logger.info("Creating optimizers...")
        self.gen_optim, self.disc_optim = self._build_optimizer(self.gen, self.disc)

        # build loss
        self.logger.info("Creating losses...")
        self._build_criterion()

        self.logger.info("Creating metric trackers...")
        self._build_metrics()

    def _build_dataloader(self):
        train_dataloader = CartoonDataLoader(
            data_dir=self.config.data_dir,
            src_style='real',
            tar_style=self.config.tar_style,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            num_workers=self.config.num_workers)
        valid_dataloader = train_dataloader.split_validation()
        return train_dataloader, valid_dataloader

    def _build_model(self):
        """ build generator and discriminator model """
        gen = Generator(self.config.image_size, self.config.down_size, self.config.num_res, self.config.skip_conn)
        disc = Discriminator(self.config.image_size, self.config.down_size)
        return gen, disc

    def _build_optimizer(self, gen, disc):
        """ build generator and discriminator optimizers """
        gen_optim = torch.optim.AdamW(
            gen.parameters(),
            lr=self.config.g_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.5, 0.999))
        disc_optim = torch.optim.AdamW(
            disc.parameters(),
            lr=self.config.d_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.5, 0.999))
        return gen_optim, disc_optim

    def _build_criterion(self):
        self.adv_criterion = eval('{}Loss'.format(self.config.adv_criterion))()
        # TODO add extra criterion you need here

    def _build_metrics(self):
        # TODO: add the loss you want to log here
        self.metric_names = ['disc', 'gen']
        self.train_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)
        self.valid_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.gen.train()
        self.disc.train()
        self.train_metrics.reset()

        for batch_idx, (src_imgs, tar_imgs) in enumerate(self.train_dataloader):
            src_imgs, tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device)
            self.gen_optim.zero_grad()
            self.disc_optim.zero_grad()

            raise NotImplementedError

            # ============ Generation ============ #


            # ============ train D ============ #


            # ============ train G ============ #

            # ============ log ============ #

            self.writer.set_step((epoch - 1) * len(self.train_dataloader) + batch_idx)
            # TODO: add the loss you want to log here

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {:d} {:d} Disc. Loss: {:.4f} Gen. Loss {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    disc_loss.item(),
                    gen_loss.item()))

        log = self.train_metrics.result()
        val_log = self._valid_epoch(epoch)
        log.update(**{'val_'+k : v for k, v in val_log.items()})
        # shuffle data loader
        self.train_dataloader.shuffle()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.gen.eval()
        self.disc.eval()

        disc_losses = []
        gen_losses = []
        self.valid_metrics.reset()
        with torch.no_grad():

            for batch_idx, (src_imgs, tar_imgs) in enumerate(self.valid_dataloader):
                src_imgs, tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device)

                # TODO similar to train but not optimizer.step()
                raise NotImplementedError
                # ============ Generation ============ #

                # ============ D Loss ============ #

                # ============ G Loss ============ #


            # log losses
            self.writer.set_step(epoch)
            self.valid_metrics.update('disc', np.mean(disc_losses))
            self.valid_metrics.update('gen', np.mean(gen_losses))

            # log images
            src_tar_imgs = torch.cat([src_imgs.cpu(), fake_tar_imgs.cpu()], dim=-1)
            self.writer.add_image('src2tar', make_grid(src_tar_imgs.cpu(), nrow=1, normalize=True))
        return self.valid_metrics.result()


    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'gen_state_dict': self.gen.state_dict() if len(self.device_ids) <= 1 else self.gen.module.state_dict(),
            'disc_state_dict': self.disc.state_dict() if len(self.device_ids) <= 1 else self.disc.module.state_dict(),
            'gen_optim': self.gen_optim.state_dict(),
            'disc_optim': self.disc_optim.state_dict()
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
        self.gen.load_state_dict(checkpoint['gen_state_dict'])
        self.disc.load_state_dict(checkpoint['disc_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.gen_optim.load_state_dict(checkpoint['gen_optim'])
        self.disc_optim.load_state_dict(checkpoint['disc_optim'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
