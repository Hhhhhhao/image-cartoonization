import torch
import torchvision
import numpy as np
from base import BaseTrainer
from models import StarGenerator, StarDiscriminator, MappingNetwork, StyleEncoder, PatchSampleF
from losses import *
from data_loaders import StarCartoonDataLoader, DiffAugment
from utils import MetricTracker


class StarCartoonTrainer(BaseTrainer):
    def __init__(self, config):
        super(StarCartoonTrainer, self).__init__(config)

        self.logger.info("Creating data loaders...")
        self.train_dataloader, self.valid_dataloader = self._build_dataloader()
        self.log_step = int(np.sqrt(self.train_dataloader.batch_size))

        self.logger.info("Creating model architecture...")
        gen, disc, map_net, samp_net = self._build_model()

        # resume
        if self.config.resume is not None:
            self._resume_checkpoint(config.resume)

        # move to device
        self.gen = gen.to(self.device)
        self.disc = disc.to(self.device)
        self.map_net = map_net.to(self.device)
        self.samp_net = samp_net.to(self.device)

        if len(self.device_ids) > 1:
            self.gen = torch.nn.DataParallel(self.gen, device_ids=self.device_ids)
            self.disc = torch.nn.DataParallel(self.disc, device_ids=self.device_ids)
            self.map_net = torch.nn.DataParallel(self.map_net, device_ids=self.device_ids)
            self.samp_net = torch.nn.DataParallel(self.samp_net, device_ids=self.device_ids)

        # optimizer
        self.logger.info("Creating optimizers...")
        self.gen_optim, self.disc_optim = self._build_optimizer(self.gen, self.disc, self.map_net, self.samp_net)

        # build loss
        self.logger.info("Creating losses...")
        self._build_criterion()

        # metric tracker
        self.logger.info("Creating metric trackers...")
        self._build_metrics()

    def _build_dataloader(self):
        train_dataloader = StarCartoonDataLoader(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            num_workers=self.config.num_workers)
        valid_dataloader = train_dataloader.split_validation()
        return train_dataloader, valid_dataloader

    def _build_model(self):
        gen = StarGenerator(self.config.image_size, self.config.down_size, self.config.num_res, self.config.skip_conn, self.config.style_size)
        disc = StarDiscriminator(self.config.image_size, self.config.down_size, num_domains=4)
        map_net = MappingNetwork(latent_dim=16, style_dim=self.config.style_size, num_domains=4)
        samp_net = PatchSampleF(use_mlp=True, gpu_ids=self.device_ids)
        return gen, disc, map_net, samp_net

    def _build_optimizer(self, gen, disc, map_net, samp_net):
        gen_optim = torch.optim.AdamW(list(gen.parameters()) + list(map_net.parameters()) + list(samp_net.parameters()),  lr=self.config.g_lr, weight_decay=self.config.weight_decay, betas=(0.5, 0.999))
        disc_optim = torch.optim.AdamW(disc.parameters(), lr=self.config.d_lr, weight_decay=self.config.weight_decay, betas=(0.5, 0.999))
        return gen_optim, disc_optim

    def _build_criterion(self):
        self.adv_loss = eval('{}Loss'.format(self.config.adv_criterion))()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.rec_loss = PatchNCELoss(self.config.nce_t).to(self.device)

    def _build_metrics(self):
        self.metric_names = ['disc', 'disc_cls', 'disc_adv',
                             'gen', 'gen_cls', 'gen_adv', 'gen_ds', 'gen_rec']
        self.train_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)
        self.valid_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)

    def _train_epoch(self, epoch):

        self.gen.train()
        self.disc.train()
        self.map_net.train()
        self.train_metrics.reset()

        for batch_idx, (src_imgs, tar_imgs, tar_labels) in enumerate(self.train_dataloader):
            src_imgs, tar_imgs, tar_labels = src_imgs.to(self.device), tar_imgs.to(self.device), tar_labels.to(self.device)
            self.gen_optim.zero_grad()
            self.disc_optim.zero_grad()
            batch_size = src_imgs.size(0)

            # generation
            tar_z = torch.randn((batch_size, self.config.latent_size)).to(self.device)
            tar_s = self.map_net(tar_z, tar_labels)
            fake_tar_imgs = self.gen(src_imgs, tar_s)

            # train G
            self.set_requires_grad(self.disc, requires_grad=False)

            # adv loss
            disc_fake_tar_logits1, disc_fake_tar_logits2 = self.disc(DiffAugment(fake_tar_imgs, policy=self.config.data_aug_policy))
            gen_adv_loss = self.adv_loss(disc_fake_tar_logits1, real=True)
            gen_cls_loss = self.cls_loss(disc_fake_tar_logits2, tar_labels)

            # diversity sensitive loss
            tar_z2 = torch.randn((batch_size, self.config.latent_size)).to(self.device)
            tar_s2 = self.map_net(tar_z2, tar_labels)
            fake_tar_imgs2 = self.gen(src_imgs, tar_s2)
            fake_tar_imgs2 = fake_tar_imgs2.detach()
            gen_ds_loss = torch.mean(torch.abs(fake_tar_imgs - fake_tar_imgs2))

            # content loss
            feat_q = self.gen.forward_encoder(src_imgs)
            feat_k = self.gen.forward_encoder(fake_tar_imgs)
            feat_k_pool, sample_ids = self.samp_net(feat_k, 128, None)
            feat_q_pool, _ = self.samp_net(feat_q, 128, sample_ids)
            gen_rec_loss = 0.0
            for f_q, f_k in zip(feat_q_pool, feat_k_pool):
                gen_rec_loss += self.rec_loss(f_q, f_k).mean()
            
            # total loss
            gen_loss = self.config.lambda_adv *  gen_adv_loss +  self.config.lambda_cls * gen_cls_loss + self.config.lambda_rec * gen_rec_loss - self.config.lambda_ds * gen_ds_loss
            gen_loss.backward()
            self.gen_optim.step()

            # train D
            self.set_requires_grad(self.disc, requires_grad=True)
            disc_real_logits1, disc_real_logits2 = self.disc(DiffAugment(tar_imgs, policy=self.config.data_aug_policy))
            disc_fake_logits1, disc_fake_logits2 = self.disc(DiffAugment(fake_tar_imgs.detach(), policy=self.config.data_aug_policy))

            # compute loss
            fake_tar_labels = torch.LongTensor(np.random.randint(0, 4, tar_imgs.size(0))).to(self.device)
            disc_adv_loss = self.adv_loss(disc_real_logits1, real=True) + self.adv_loss(disc_fake_logits1, real=False)
            disc_cls_loss = self.cls_loss(disc_real_logits2, tar_labels) + self.cls_loss(disc_real_logits2, fake_tar_labels)
            disc_loss = self.config.lambda_adv * disc_adv_loss + self.config.lambda_cls * disc_cls_loss
            disc_loss.backward()
            self.disc_optim.step()

            # ============ log ============ #
            self.writer.set_step((epoch - 1) * len(self.train_dataloader) + batch_idx)
            self.train_metrics.update('disc', disc_loss.item())
            self.train_metrics.update('disc_cls', disc_cls_loss.item())
            self.train_metrics.update('disc_adv', disc_adv_loss.item())
            self.train_metrics.update('gen', gen_loss.item())
            self.train_metrics.update('gen_adv', gen_adv_loss.item())
            self.train_metrics.update('gen_cls', gen_cls_loss.item())
            self.train_metrics.update('gen_ds', gen_ds_loss.item())
            self.train_metrics.update('gen_rec', gen_rec_loss.item())

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
        self.map_net.eval()

        disc_losses = []
        disc_adv_losses = []
        disc_cls_losses = []
        gen_losses = []
        gen_adv_losses = []
        gen_cls_losses = []
        gen_ds_losses = []
        gen_rec_losses = []

        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (src_imgs, tar_imgs, tar_labels) in enumerate(self.valid_dataloader):
                src_imgs, tar_imgs, tar_labels = src_imgs.to(self.device), tar_imgs.to(self.device), tar_labels.to(self.device)
                batch_size = src_imgs.size(0)

                # generation
                tar_z = torch.randn((batch_size, self.config.latent_size)).to(self.device)
                tar_s = self.map_net(tar_z, tar_labels)
                fake_tar_imgs = self.gen(src_imgs, tar_s)

                # adv loss
                disc_fake_tar_logits1, disc_fake_tar_logits2 = self.disc(
                    DiffAugment(fake_tar_imgs, policy=self.config.data_aug_policy))
                gen_adv_loss = self.adv_loss(disc_fake_tar_logits1, real=True)
                gen_cls_loss = self.cls_loss(disc_fake_tar_logits2, tar_labels)

                # diversity sensitive loss
                tar_z2 = torch.randn((batch_size, self.config.latent_size)).to(self.device)
                tar_s2 = self.map_net(tar_z2, tar_labels)
                fake_tar_imgs2 = self.gen(src_imgs, tar_s2)
                fake_tar_imgs2 = fake_tar_imgs2.detach()
                gen_ds_loss = torch.mean(torch.abs(fake_tar_imgs - fake_tar_imgs2))

                # content loss
                feat_q, _ = self.gen.forward_encoder(src_imgs)
                feat_k, _ = self.gen.forward_encoder(fake_tar_imgs)
                gen_rec_loss = self.rec_loss(feat_q, feat_k)

                # total loss
                gen_loss = self.config.lambda_adv * gen_adv_loss + self.config.lambda_cls * gen_cls_loss + self.config.lambda_rec * gen_rec_loss - self.config.lambda_ds * gen_ds_loss

                # train D
                self.set_requires_grad(self.disc, requires_grad=True)
                disc_real_logits1, disc_real_logits2 = self.disc(tar_imgs)
                disc_fake_logits1, _ = self.disc(fake_tar_imgs.detach())
                _, disc_tar_logits2 = self.disc(src_imgs)

                # compute loss
                disc_adv_loss = self.adv_loss(disc_real_logits1, real=True) + self.adv_loss(disc_fake_logits1,
                                                                                            real=False)
                disc_cls_loss = self.cls_loss(disc_real_logits2, tar_labels)
                disc_loss = self.config.lambda_adv * disc_adv_loss + self.config.lambda_cls * disc_cls_loss

                disc_losses.append(disc_loss.item())
                disc_adv_losses.append(disc_adv_loss.item())
                disc_cls_losses.append(disc_cls_loss.item())
                gen_losses.append(gen_loss.item())
                gen_adv_losses.append(gen_adv_loss.item())
                gen_cls_losses.append(gen_cls_loss.item())
                gen_ds_losses.append(gen_ds_loss.item())
                gen_rec_losses.append(gen_rec_loss.item())

            # log losses
            self.writer.set_step(epoch)
            self.valid_metrics.update('disc', np.mean(disc_losses))
            self.valid_metrics.update('disc_adv', np.mean(disc_adv_losses))
            self.valid_metrics.update('disc_cls', np.mean(disc_cls_losses))
            self.valid_metrics.update('gen', np.mean(gen_losses))
            self.valid_metrics.update('gen_adv', np.mean(gen_adv_losses))
            self.valid_metrics.update('gen_ds', np.mean(gen_ds_losses))
            self.valid_metrics.update('gen_rec', np.mean(gen_rec_losses))

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
            'map_state_dict': self.map_net.state_dict() if len(
                self.device_ids) <= 1 else self.map_net.module.state_dict(),
            'gen_optim': self.gen_optim.state_dict(),
            'disc_optim': self.disc_optim.state_dict(),
            'map_optim': self.map_net_optim.state_dict()
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
        self.map_net.load_state_dict(checkpoint['map_state_dict'])
        # self.style_enc.load_state_dict(checkpoint['sty_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.gen_optim.load_state_dict(checkpoint['gen_optim'])
        self.disc_optim.load_state_dict(checkpoint['disc_optim'])
        self.map_net_optim.load_state_dict(checkpoint['map_optim'])
        # self.style_enc_optim.load_state_dict(checkpoint['sty_optim'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))