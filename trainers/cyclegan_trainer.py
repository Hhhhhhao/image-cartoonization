import torch
from torchvision.utils import make_grid
import numpy as np
from base import BaseTrainer
from models import Generator, Discriminator
from losses import *
from data_loaders import CartoonDataLoader
from utils import MetricTracker


class CycleGANTrainer(BaseTrainer):
    def __init__(self, config):
        super(CycleGANTrainer, self).__init__(config)

        self.logger.info("Creating data loaders...")
        self.train_dataloader, self.valid_dataloader = self._build_dataloader()
        self.log_step = int(np.sqrt(self.train_dataloader.batch_size))

        self.logger.info("Creating model architecture...")
        gen_src_tar, gen_tar_src, disc_src, disc_tar = self._build_model()
        # resume
        if self.config.resume is not None:
            self._resume_checkpoint(config.resume)
        # move to device
        self.gen_src_tar = gen_src_tar.to(self.device)
        self.gen_tar_src = gen_tar_src.to(self.device)
        self.disc_src = disc_src.to(self.device)
        self.disc_tar = disc_tar.to(self.device)
        if len(self.device_ids) > 1:
            self.gen_src_tar = torch.nn.DataParallel(self.gen_src_tar, device_ids=self.device_ids)
            self.disc_src = torch.nn.DataParallel(self.disc_src, device_ids=self.device_ids)
            self.gen_tar_src = torch.nn.DataParallel(self.gen_tar_src, device_ids=self.device_ids)
            self.disc_tar = torch.nn.DataParallel(self.disc_tar, device_ids=self.device_ids)

        self.logger.info("Creating optimizers...")
        self.gen_optim, self.disc_optim = self._build_optimizer(self.gen_src_tar, self.gen_tar_src, self.disc_src, self.disc_tar)

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
        gen_src_tar = Generator(self.config.image_size, self.config.down_size, self.config.num_res, self.config.skip_conn)
        gen_tar_src = Generator(self.config.image_size, self.config.down_size, self.config.num_res, self.config.skip_conn)
        disc_src = Discriminator(self.config.image_size, self.config.down_size)
        disc_tar = Discriminator(self.config.image_size, self.config.down_size)
        return gen_src_tar, gen_tar_src, disc_src, disc_tar

    def _build_optimizer(self, gen_src_tar, gen_tar_src, disc_src, disc_tar):
        """ build generator and discriminator optimizers """
        gen_optim = torch.optim.AdamW(
            list(gen_src_tar.parameters()) + list(gen_tar_src.parameters()),
            lr=self.config.g_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.5, 0.999))
        disc_optim = torch.optim.AdamW(
            list(disc_src.parameters()) + list(disc_tar.parameters()),
            lr=self.config.d_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.5, 0.999))
        return gen_optim, disc_optim

    def _build_criterion(self):
        self.adv_criterion = eval('{}Loss'.format(self.config.adv_criterion))()
        self.cyc_criterion = torch.nn.L1Loss()
        self.ide_criterion = torch.nn.L1Loss()

    def _build_metrics(self):
        self.metric_names = ['disc_src', 'disc_tar', 'gen_src_tar', 'gen_tar_src']
        self.train_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)
        self.valid_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.gen_src_tar.train()
        self.gen_tar_src.train()
        self.disc_src.train()
        self.disc_tar.train()
        self.train_metrics.reset()

        for batch_idx, (src_imgs, tar_imgs) in enumerate(self.train_dataloader):
            src_imgs, tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device)
            self.gen_optim.zero_grad()
            self.disc_optim.zero_grad()

            # ============ generation ============ #
            fake_tar_imgs = self.gen_src_tar(src_imgs)
            fake_src_imgs = self.gen_tar_src(tar_imgs)

            # ============ train G ============ #
            self.set_requires_grad([self.disc_tar, self.disc_src], requires_grad=False)

            # discriminator loss
            disc_fake_src_logits = self.disc_src(fake_src_imgs)
            disc_fake_tar_logits = self.disc_tar(fake_tar_imgs)
            disc_src_loss_ = self.adv_criterion(disc_fake_src_logits, real=True)
            disc_tar_loss_ = self.adv_criterion(disc_fake_tar_logits, real=True)

            # translate back and cycle consistant loss
            rec_src_imgs = self.gen_tar_src(fake_tar_imgs)
            rec_tar_imgs = self.gen_src_tar(fake_src_imgs)
            rec_src_loss = self.cyc_criterion(rec_src_imgs, src_imgs)
            rec_tar_loss = self.cyc_criterion(rec_tar_imgs, tar_imgs)

            # identity loss
            idt_tar_imgs = self.gen_src_tar(tar_imgs)
            idt_src_imgs = self.gen_tar_src(src_imgs)

            # total generator loss
            gen_src_loss = self.config.lambda_adv * disc_tar_loss_ + self.config.lambda_rec * rec_src_loss + 0.5 * self.config.lambda_rec * idt_tar_imgs
            gen_tar_loss = self.config.lambda_adv * disc_src_loss_ + self.config.lambda_rec * rec_tar_loss + 0.5 * self.config.lambda_rec * idt_src_imgs
            gen_loss = gen_src_loss + gen_tar_loss
            gen_loss.backward()
            self.gen_optim.step()

            # ============ train D ============ #
            self.set_requires_grad([self.disc_tar, self.disc_src], requires_grad=True)

            # get logits from discriminators
            disc_src_real_logits = self.disc_src(src_imgs)
            disc_src_fake_logits = self.disc_src(fake_src_imgs.detach())
            disc_tar_real_logits = self.disc_tar(tar_imgs)
            disc_tar_fake_logits = self.disc_tar(fake_tar_imgs.detach())

            # compute loss
            disc_src_loss = self.adv_criterion(disc_src_real_logits, real=True) + self.adv_criterion(disc_src_fake_logits, real=False)
            disc_tar_loss = self.adv_criterion(disc_tar_real_logits, real=True) + self.adv_criterion(disc_tar_fake_logits, real=False)
            disc_loss = disc_src_loss + disc_tar_loss
            disc_loss.backward()
            self.disc_optim.step()

            # ============ log ============ #
            self.writer.set_step((epoch - 1) * len(self.train_dataloader) + batch_idx)
            self.train_metrics.update('disc_src', disc_src_loss.item())
            self.train_metrics.update('disc_tar', disc_tar_loss.item())
            self.train_metrics.update('gen_src_tar', gen_src_loss.item())
            self.train_metrics.update('gen_tar_src', gen_tar_loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {:d} {:s} Disc. Loss: {:.4f} Gen. Loss {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    disc_loss.item(),
                    gen_loss.item()))

        log = self.train_metrics.result()
        val_log = self._valid_epoch(epoch)
        log.update(**{'val_'+k : v for k, v in val_log.items()})
        # shuffle data loader
        self.train_dataloader.shuffle_dataset()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.gen_src_tar.eval()
        self.gen_tar_src.eval()
        self.disc_src.eval()
        self.disc_tar.eval()

        disc_src_losses = []
        disc_tar_losses = []
        gen_src_tar_losses = []
        gen_tar_src_losses = []
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (src_imgs, tar_imgs) in enumerate(self.valid_dataloader):
                src_imgs, tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device)

                # ============ Generation ============ #
                fake_tar_imgs = self.gen_src_tar(src_imgs)
                fake_src_imgs = self.gen_tar_src(tar_imgs)

                # ============ G Loss ============ #

                # discriminator loss
                disc_fake_src_logits = self.disc_src(fake_src_imgs)
                disc_fake_tar_logits = self.disc_tar(fake_tar_imgs)
                disc_src_loss_ = self.adv_criterion(disc_fake_src_logits, real=True)
                disc_tar_loss_ = self.adv_criterion(disc_fake_tar_logits, real=True)

                # translate back and cycle consistant loss
                rec_src_imgs = self.gen_tar_src(fake_tar_imgs)
                rec_tar_imgs = self.gen_src_tar(fake_src_imgs)
                rec_src_loss = self.cyc_criterion(rec_src_imgs, src_imgs)
                rec_tar_loss = self.cyc_criterion(rec_tar_imgs, tar_imgs)

                gen_src_loss = self.config.lambda_adv * disc_src_loss_ + self.config.lambda_rec * rec_src_loss
                gen_tar_loss = self.config.lambda_adv * disc_tar_loss_ + self.config.lambda_rec * rec_tar_loss

                # ============ D Loss ============ #

                # get logits from discriminators
                disc_src_real_logits = self.disc_src(src_imgs)
                disc_src_fake_logits = self.disc_src(fake_src_imgs.detach())
                disc_tar_real_logits = self.disc_tar(tar_imgs)
                disc_tar_fake_logits = self.disc_tar(fake_tar_imgs.detach())

                # compute loss
                disc_src_loss = self.adv_criterion(disc_src_real_logits, real=True) + self.adv_criterion(
                    disc_src_fake_logits, real=False)
                disc_tar_loss = self.adv_criterion(disc_tar_real_logits, real=True) + self.adv_criterion(
                    disc_tar_fake_logits, real=False)


                disc_src_losses.append(disc_src_loss.item())
                disc_tar_losses.append(disc_tar_loss.item())
                gen_src_tar_losses.append(gen_src_loss.item())
                gen_tar_src_losses.append(gen_tar_loss.item())

            # log losses
            self.writer.set_step(epoch)
            self.valid_metrics.update('disc_src', np.mean(disc_src_losses))
            self.valid_metrics.update('disc_tar', np.mean(disc_tar_losses))
            self.valid_metrics.update('gen_src_tar', np.mean(gen_src_tar_losses))
            self.valid_metrics.update('gen_tar_src', np.mean(gen_tar_src_losses))

            # log images
            src_tar_imgs = torch.cat([src_imgs.cpu(), fake_tar_imgs.cpu()], dim=-1)
            self.writer.add_image('src2tar', make_grid(src_tar_imgs.cpu(), nrow=1, normalize=True))
            tar_src_imgs = torch.cat([tar_imgs.cpu(), fake_src_imgs.cpu()], dim=-1)
            self.writer.add_image('tar2src', make_grid(tar_src_imgs.cpu(), nrow=1, normalize=True))
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
            'gen_src_tar_state_dict': self.gen_src_tar.state_dict() if len(self.device_ids) <= 1 else self.gen_src_tar.module.state_dict(),
            'gen_tar_src_state_dict': self.gen_tar_src.state_dict() if len(self.device_ids) <= 1 else self.gen_tar_src.module.state_dict(),
            'disc_src_state_dict': self.disc_src.state_dict() if len(self.device_ids) <= 1 else self.disc_src.module.state_dict(),
            'disc_tar_state_dict': self.disc_tar.state_dict() if len(self.device_ids) <= 1 else self.disc_tar.module.state_dict(),
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
        self.gen_src_tar.load_state_dict(checkpoint['gen_src_tar_state_dict'])
        self.gen_tar_src.load_state_dict(checkpoint['gen_tar_src_state_dict'])
        self.disc_src.load_state_dict(checkpoint['disc_src_state_dict'])
        self.disc_tar.load_state_dict(checkpoint['disc_tar_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.gen_optim.load_state_dict(checkpoint['gen_optim'])
        self.disc_optim.load_state_dict(checkpoint['disc_optim'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
