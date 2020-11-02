import torch
import torchvision
import numpy as np
from base import BaseTrainer
from models import Generator, Discriminator
from losses import *
from data_loaders import CartoonGANDataLoader
from utils import MetricTracker


class CartoonGanTrainer(BaseTrainer):
    def __init__(self, config):
        super(CartoonGanTrainer, self).__init__(config)

        self.logger.info("Creating data loaders...")
        self.train_dataloader, self.valid_dataloader = self._build_dataloader()
        self.log_step = int(np.sqrt(self.train_dataloader.batch_size))

        self.logger.info("Creating model architecture...")
        g,d = self._build_model()

        # resume
        if self.config.resume is not None:
            self._resume_checkpoint(config.resume)

        # move to device
        self.gen = g.to(self.device)
        self.disc = d.to(self.device)
        if len(self.device_ids) > 1:
            self.gen = torch.nn.DataParallel(self.gen, device_ids=self.device_ids)
            self.disc = torch.nn.DataParallel(self.disc, device_ids=self.device_ids)

        # optimizer
        self.logger.info("Creating optimizers...")
        self.gen_optim, self.disc_optim = self._build_optimizer(self.gen, self.disc)

        # build loss
        self.logger.info("Creating losses...")
        self._build_criterion()

        # metric tracker
        self.logger.info("Creating metric trackers...")
        self._build_metrics()

    def _build_dataloader(self):
        train_dataloader = CartoonGANDataLoader(
            data_dir=self.config.data_dir,
            src_style='real',
            tar_style=self.config.tar_style,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            num_workers=self.config.num_workers)
        valid_dataloader = train_dataloader.split_validation()
        return train_dataloader, valid_dataloader

    def _build_model(self):
        gen = Generator(self.config.image_size, self.config.down_size, self.config.num_res, self.config.skip_conn)
        disc = Discriminator(self.config.image_size, self.config.down_size)
        return gen,disc

    def _build_optimizer(self, gen,disc):
        gen_optim = torch.optim.AdamW(gen.parameters(),  lr=self.config.g_lr, weight_decay=self.config.weight_decay, betas=(0.5, 0.999))
        disc_optim = torch.optim.AdamW(disc.parameters(), lr=self.config.g_lr, weight_decay=self.config.weight_decay, betas=(0.5, 0.999))
        return gen_optim, disc_optim

    def _build_criterion(self):
        self.adv_loss = eval('{}Loss'.format(self.config.adv_criterion))()
        self.cont_loss = VGGPerceptualLoss().to(self.device)

    def _build_metrics(self):
        self.metric_names = ['disc', 'gen']
        self.train_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)
        self.valid_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)

    def _train_epoch(self, epoch):

        self.gen.train()
        self.disc.train()
        self.train_metrics.reset()

        for batch_idx, (src_imgs, tar_imgs, smooth_tar_imgs) in enumerate(self.train_dataloader):
            src_imgs, tar_imgs, smooth_tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device), smooth_tar_imgs.to(self.device)
            self.gen_optim.zero_grad()
            self.disc_optim.zero_grad()

            # generation
            fake_tar_imgs = self.gen(src_imgs)

            # train G
            self.set_requires_grad(self.disc, requires_grad=False)
            disc_fake_tar_logits = self.disc(fake_tar_imgs)
            gen_adv_loss = self.adv_loss(disc_fake_tar_logits, real=True)
            gen_cont_loss = self.cont_loss(fake_tar_imgs, src_imgs)
            gen_loss = self.config.lambda_adv * gen_adv_loss +  self.config.lambda_rec * gen_cont_loss
            gen_loss.backward()
            self.gen_optim.step()

            # train D
            self.set_requires_grad(self.disc, requires_grad=True)
            disc_real_logits = self.disc(tar_imgs)
            disc_fake_logits = self.disc(fake_tar_imgs.detach())
            disc_edge_logits = self.disc(smooth_tar_imgs)

            # compute loss
            disc_loss = self.adv_loss(disc_real_logits, real=True) + self.adv_loss(disc_fake_logits, real=False) + self.adv_loss(disc_edge_logits, real=True)
            disc_loss.backward()
            self.disc_optim.step()

            # ============ log ============ #
            self.writer.set_step((epoch - 1) * len(self.train_dataloader) + batch_idx)
            self.train_metrics.update('disc', disc_loss.item())
            self.train_metrics.update('gen', gen_loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {:d} {:s} Disc. Loss: {:.4f} Gen. Loss {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    disc_loss.item(),
                    gen_loss.item()))

        log = self.train_metrics.result()
        val_log = self._valid_epoch(epoch)
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        # shuffle data loader
        self.train_dataloader.shuffle_dataset()
        return log

    def _valid_epoch(self,epoch):
        self.gen.eval()
        self.disc.eval()
        disc_losses = []
        gen_losses = []
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (src_imgs, tar_imgs, smooth_tar_imgs) in enumerate(self.valid_dataloader):
                src_imgs, tar_imgs, smooth_tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device), smooth_tar_imgs.to(self.device)

                # generation
                fake_tar_imgs = self.gen(src_imgs)

                # D loss
                disc_fake_logits = self.disc(fake_tar_imgs.detach())
                disc_real_logits = self.disc(tar_imgs)
                disc_edge_logits = self.disc(smooth_tar_imgs)
                disc_loss = self.adv_loss(disc_real_logits, real=True) + self.adv_loss(disc_fake_logits, real=False) + self.adv_loss(disc_edge_logits, real=True)

                # G loss
                disc_fake_tar_logits = self.disc(fake_tar_imgs)
                gen_disc_loss = self.adv_loss(disc_fake_tar_logits, real=True)
                gen_content_loss = self.cont_loss(fake_tar_imgs, src_imgs)
                gen_loss = self.config.lambda_adv * gen_disc_loss +  self.config.lambda_rec * gen_content_loss

                disc_losses.append(disc_loss.item())
                gen_losses.append(gen_loss.item())

            # log losses
            self.writer.set_step(epoch)
            self.valid_metrics.update('disc', np.mean(disc_losses))
            self.valid_metrics.update('gen', np.mean(gen_losses))

            # log images
            src_tar_imgs = torch.cat([src_imgs.cpu(), fake_tar_imgs.cpu()], dim=-1)
            self.writer.add_image('src2tar', torchvision.utils.make_grid(src_tar_imgs.cpu(), nrow=1, normalize=True))

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
            'gen_state_dict': self.gen.state_dict() if len(
                self.device_ids) <= 1 else self.gen.module.state_dict(),
            'disc_state_dict': self.disc.state_dict() if len(
                self.device_ids) <= 1 else self.disc.module.state_dict(),
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