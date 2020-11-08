import torch
from torchvision.utils import make_grid
import numpy as np
from base import BaseTrainer
from models import Generator, Discriminator
from losses import *
from data_loaders import CartoonDataLoader, DiffAugment
from utils import MetricTracker, guided_filter, color_shift, superpixel


class WhiteboxTrainer(BaseTrainer):
    def __init__(self, config):
        super(WhiteboxTrainer, self).__init__(config)

        self.logger.info("Creating data loaders...")
        self.train_dataloader, self.valid_dataloader = self._build_dataloader()
        self.log_step = int(np.sqrt(self.train_dataloader.batch_size))

        self.logger.info("Creating model architecture...")
        gen, disc_blur, disc_gray = self._build_model()
        # resume
        if self.config.resume is not None:
            self._resume_checkpoint(config.resume)
        # move to device
        self.gen = gen.to(self.device)
        self.disc_blur = disc_blur.to(self.device)
        self.disc_gray = disc_gray.to(self.device)
        if len(self.device_ids) > 1:
            self.gen = torch.nn.DataParallel(self.gen, device_ids=self.device_ids)
            self.disc_blur = torch.nn.DataParallel(self.disc_blur, device_ids=self.device_ids)
            self.disc_gray = torch.nn.DataParallel(self.disc_gray, device_ids=self.device_ids)

        self.logger.info("Creating optimizers...")
        self.gen_optim, self.disc_blur_optim ,self.disc_gray_optim = self._build_optimizer(self.gen, self.disc_blur, self.disc_gray)

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
        disc_blur = Discriminator(self.config.image_size, self.config.down_size)
        disc_gray = Discriminator(self.config.image_size, self.config.down_size)
        return gen, disc_blur, disc_gray

    def _build_optimizer(self, gen, disc_blur, disc_gray):
        """ build generator and discriminator optimizers """
        gen_optim = torch.optim.AdamW(
            gen.parameters(),
            lr=self.config.g_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.5, 0.999))
        disc_blur_optim = torch.optim.AdamW(
            disc_blur.parameters(),
            lr=self.config.d_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.5, 0.999))
        disc_gray_optim = torch.optim.AdamW(
            disc_gray.parameters(),
            lr=self.config.d_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.5, 0.999))
        return gen_optim, disc_blur_optim, disc_gray_optim

    def _build_criterion(self):
        self.adv_criterion = eval('{}Loss'.format(self.config.adv_criterion))()
        self.tv_loss = TVLoss()
        self.vgg_loss = VGGPerceptualLoss().to(self.device)

    def _build_metrics(self):
        self.metric_names = ['disc', 'gen', 'disc_blur_loss', 'disc_gray_loss', 'gen_blur_loss', 'gen_gray_loss', 'gen_recon_loss', 'gen_tv_loss']
        self.train_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)
        self.valid_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.gen.train()
        self.disc_blur.train()
        self.disc_gray.train()
        self.train_metrics.reset()

        for batch_idx, (src_imgs, tar_imgs) in enumerate(self.train_dataloader):
            src_imgs, tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device)
            self.gen_optim.zero_grad()
            self.disc_blur_optim.zero_grad()
            self.disc_gray_optim.zero_grad()

            # ============ Generation ============ #
            fake_tar_imgs = self.gen(src_imgs)
            fake_tar_imgs = guided_filter(src_imgs, fake_tar_imgs, r=1)

            # ============ train G ============ #
            self.set_requires_grad(self.disc_gray, requires_grad=False)
            self.set_requires_grad(self.disc_blur, requires_grad=False)
            tv_loss = self.tv_loss(fake_tar_imgs)

            # surface representation
            blur_fake_tar = guided_filter(fake_tar_imgs, fake_tar_imgs, r=5, eps=2e-1)
            disc_blur_fake_logits = self.disc_blur(DiffAugment(blur_fake_tar, policy=self.config.data_aug_policy))

            # texture representation
            gray_fake_tar = color_shift(fake_tar_imgs)
            disc_gray_fake_logits = self.disc_gray(DiffAugment(gray_fake_tar, policy=self.config.data_aug_policy))

            # surface and texture loss
            gen_surface_loss = self.adv_criterion(disc_blur_fake_logits, real=True)
            gen_texture_loss = self.adv_criterion(disc_gray_fake_logits, real=True)

            # structure loss
            content_loss = self.vgg_loss(fake_tar_imgs, src_imgs)

            total_gen = tv_loss + 1e-1 * gen_surface_loss + gen_texture_loss + 10 * content_loss
            total_gen.backward()
            self.gen_optim.step()

            # ============ train D ============ #
            self.set_requires_grad(self.disc_gray, requires_grad=True)
            self.set_requires_grad(self.disc_blur, requires_grad=True)

            # surface representation
            blur_fake_tar = guided_filter(fake_tar_imgs.detach(), fake_tar_imgs.detach(), r=5, eps=2e-1)
            blur_real_tar = guided_filter(tar_imgs, tar_imgs, r=5, eps=2e-1)
            blur_fake_logits = self.disc_blur(DiffAugment(blur_fake_tar, policy=self.config.data_aug_policy))
            blur_real_logits = self.disc_blur(DiffAugment(blur_real_tar, policy=self.config.data_aug_policy))

            # texture representation
            gray_fake_tar = color_shift(fake_tar_imgs.detach())
            gray_real_tar = color_shift(tar_imgs)
            gray_fake_logits = self.disc_gray(DiffAugment(gray_fake_tar, policy=self.config.data_aug_policy))
            gray_real_logits = self.disc_gray(DiffAugment(gray_real_tar, policy=self.config.data_aug_policy))

            disc_blur_loss = self.adv_criterion(blur_real_logits, real=True) + self.adv_criterion(blur_fake_logits, real=False)
            disc_gray_loss = self.adv_criterion(gray_real_logits, real=True) + self.adv_criterion(gray_fake_logits, real=False)

            total_disc = disc_blur_loss + disc_gray_loss
            total_disc.backward()
            self.disc_blur_optim.step()
            self.disc_gray_optim.step()

            # ============ log ============ #
            self.writer.set_step((epoch - 1) * len(self.train_dataloader) + batch_idx)

            self.train_metrics.update('disc_blur_loss', disc_blur_loss.item())
            self.train_metrics.update('disc_gray_loss', disc_gray_loss.item())
            self.train_metrics.update('gen_blur_loss', gen_surface_loss.item())
            self.train_metrics.update('gen_gray_loss', gen_texture_loss.item())
            self.train_metrics.update('gen_recon_loss', content_loss.item())
            self.train_metrics.update('gen_tv_loss', tv_loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {:d} {:s} Disc. Loss: {:.4f} Gen. Loss {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    total_disc.item(),
                    total_gen.item()))

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
        self.gen.eval()
        self.disc_blur.eval()
        self.disc_gray.eval()

        disc_blur_losses = []
        disc_gray_losses = []
        gen_blur_losses = []
        gen_gray_losses = []
        gen_recon_losses = []
        gen_tv_losses = []
        gen_losses = []
        disc_losses = []

        self.valid_metrics.reset()
        with torch.no_grad():

            for batch_idx, (src_imgs, tar_imgs) in enumerate(self.valid_dataloader):
                src_imgs, tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device)

                # ============ Generation ============ #
                fake_tar_imgs = self.gen(src_imgs)

                # ============ train G ============ #
                tv_loss = self.tv_loss(fake_tar_imgs)

                # surface representation
                blur_fake_tar = guided_filter(fake_tar_imgs, fake_tar_imgs, r=5, eps=2e-1)
                disc_blur_fake_logits = self.disc_blur(DiffAugment(blur_fake_tar, policy=self.config.data_aug_policy))

                # texture representation
                gray_fake_tar = color_shift(fake_tar_imgs)
                disc_gray_fake_logits = self.disc_gray(DiffAugment(gray_fake_tar, policy=self.config.data_aug_policy))

                # surface and texture loss
                gen_surface_loss = self.adv_criterion(disc_blur_fake_logits, real=True)
                gen_texture_loss = self.adv_criterion(disc_gray_fake_logits, real=True)

                # structure loss
                content_loss = self.vgg_loss(fake_tar_imgs, src_imgs)

                total_gen = tv_loss + 1e-1 * gen_surface_loss + gen_texture_loss + content_loss

                # ============ train D ============ #

                # surface representation
                blur_fake_tar = guided_filter(fake_tar_imgs, fake_tar_imgs, r=5, eps=2e-1)
                blur_real_tar = guided_filter(tar_imgs, tar_imgs, r=5, eps=2e-1)
                blur_fake_logits = self.disc_blur(DiffAugment(blur_fake_tar, policy=self.config.data_aug_policy))
                blur_real_logits = self.disc_blur(DiffAugment(blur_real_tar, policy=self.config.data_aug_policy))

                # texture representation
                gray_fake_tar = color_shift(fake_tar_imgs)
                gray_real_tar = color_shift(tar_imgs)
                gray_fake_logits = self.disc_gray(DiffAugment(gray_fake_tar, policy=self.config.data_aug_policy))
                gray_real_logits = self.disc_gray(DiffAugment(gray_real_tar, policy=self.config.data_aug_policy))

                disc_blur_loss = self.adv_criterion(blur_real_logits, real=True) + self.adv_criterion(blur_fake_logits,
                                                                                                      real=False)
                disc_gray_loss = self.adv_criterion(gray_real_logits, real=True) + self.adv_criterion(gray_fake_logits,
                                                                                                      real=False)

                total_disc = disc_blur_loss + disc_gray_loss

                disc_blur_losses.append(disc_blur_loss.item())
                disc_gray_losses.append(disc_gray_loss.item())
                gen_blur_losses.append(gen_surface_loss.item())
                gen_gray_losses.append(gen_texture_loss.item())
                gen_recon_losses.append(content_loss.item())
                gen_tv_losses.append(tv_loss.item())
                gen_losses.append(total_gen.item())
                disc_losses.append(total_disc.item())

            # log losses
            self.writer.set_step(epoch)
            self.valid_metrics.update('disc', np.mean(gen_losses))
            self.valid_metrics.update('gen', np.mean(disc_losses))
            self.train_metrics.update('disc_blur_loss', np.mean(disc_blur_losses))
            self.train_metrics.update('disc_gray_loss', np.mean(disc_gray_losses))
            self.train_metrics.update('gen_blur_loss', np.mean(gen_blur_losses))
            self.train_metrics.update('gen_gray_loss', np.mean(gen_gray_losses))
            self.train_metrics.update('gen_recon_loss', np.mean(gen_recon_losses))
            self.train_metrics.update('gen_tv_loss', np.mean(gen_tv_losses))

            # log images
            output_imgs = torch.cat([src_imgs.cpu(), fake_tar_imgs.cpu()], dim=-1)
            self.writer.add_image('src2tar', make_grid(output_imgs.cpu(), nrow=1, normalize=True))
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
            'disc_blur_state_dict': self.disc_blur.state_dict() if len(self.device_ids) <= 1 else self.disc_blur.module.state_dict(),
            'disc_gray_state_dict': self.disc_gray.state_dict() if len(self.device_ids) <= 1 else self.disc_gray.module.state_dict(),
            'gen_optim': self.gen_optim.state_dict(),
            'disc_blur_optim': self.disc_blur_optim.state_dict(),
            'disc_gray_optim': self.disc_gray_optim.state_dict()
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
        self.disc_blur.load_state_dict(checkpoint['disc_blur_state_dict'])
        self.disc_gray.load_state_dict(checkpoint['disc_gray_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.gen_optim.load_state_dict(checkpoint['gen_optim'])
        # self.disc_optim.load_state_dict(checkpoint['disc_optim'])
        self.disc_blur_optim.load_state_dict(checkpoint['disc_blur_optim'])
        self.disc_gray_optim.load_state_dict(checkpoint['disc_gray_optim'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
