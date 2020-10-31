import torch
import torchvision
import numpy as np
from base import BaseTrainer
from models import Generator, Discriminator
from losses import *
from data_loaders import CartoonDataLoader
from utils import MetricTracker

# VGG loss, Cite from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

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
        gen = Generator(self.config.image_size, self.config.down_size, self.config.num_res, self.config.skip_conn)
        disc = Discriminator(self.config.image_size, self.config.down_size)
        return gen,disc

    def _build_optimizer(self, gen,disc):
        gen_optim = torch.optim.AdamW(gen.parameters(), lr=self.config.g_lr, weight_decay=self.config.weight_decay, betas=(0.5, 0.999))
        disc_optim = torch.optim.AdamW(disc.parameters(),lr=self.config.g_lr, weight_decay=self.config.weight_decay, betas=(0.5, 0.999))
        return gen_optim,disc_optim

    def _build_criterion(self):
        self.adv_loss = torch.nn.MSELoss().to(self.device) # not sure if needed to move the loss to device
        self.content_loss = VGGPerceptualLoss().to(self.device)

    def _build_metrics(self):
        self.metric_names = ['disc','gen']
        self.train_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)
        self.valid_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)


    def _train_epoch(self, epoch):

        self.gen.train()
        self.disc.train()
        self.train_metrics.reset()

        # not sure about this real and fake
        real = torch.ones(self.config.batch_size,1,self.config.image_size,self.config.image_size)
        fake = torch.zeros(self.config.batch_size,1,self.config.image_size,self.config.image_size)

        for batch_index,(src_imgs, tar_imgs) in enumerate(self.train_dataloader):
            src_imgs, tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device)
            self.gen_optim.zero_grad()
            self.disc_optim.zero_grad()

            # generation
            fake_tar_imgs = self.gen(src_imgs)

            # train D
            self.gen_optim.zero_grad()
            disc_tar_real_logits = self.disc(tar_imgs)
            disc_src_fake_logits = self.disc(fake_tar_imgs.detach())

            # compute loss
            disc_loss = self.adv_loss(disc_tar_real_logits,real) + self.adv_loss(disc_src_fake_logits,fake)
            # disc_loss = self.adv_loss(disc_tar_real_logits, real=True) + self.adv_loss(disc_src_fake_logits, real=False)
            disc_loss.backward()
            self.disc_optim.step()

            # train G
            gen_src_logits = self.gen(src_imgs)
            disc_fake_src_logits = self.disc(gen_src_logits)
            gen_disc_loss = self.adv_loss(disc_fake_src_logits,real)
            # gen_disc_loss = self.adv_loss(disc_fake_src_logits, real=True)
            gen_content_loss = self.content_loss(src_imgs,gen_src_logits)
            gen_loss = gen_disc_loss + gen_content_loss
            gen_loss.backward()
            self.gen_optim.step()

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
                break

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
            for batch_idx, (src_imgs, tar_imgs) in enumerate(self.valid_dataloader):
                src_imgs, tar_imgs = src_imgs.to(self.device), tar_imgs.to(self.device)

                # generation
                fake_tar_imgs = self.gen(src_imgs)

                # D loss
                disc_src_fake_logits = self.disc(fake_tar_imgs.detach())
                disc_src_real_logits = self.disc(tar_imgs)
                # disc_loss = self.adv_loss(disc_tar_real_logits, real) + self.adv_loss(disc_src_fake_logits, fake)
                disc_loss = self.adv_loss(disc_src_real_logits, real=True) + self.adv_loss(disc_src_fake_logits, real=False)

                # G loss
                disc_fake_tar_logits = self.disc(fake_tar_imgs)
                # gen_disc_loss = self.adv_loss(disc_fake_tar_logits, real)
                gen_disc_loss = self.adv_loss(disc_fake_tar_logits, real=True)
                gen_content_loss = self.content_loss(src_imgs, fake_tar_imgs)
                gen_loss = gen_disc_loss + gen_content_loss

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

if __name__ == "__main__":
    # test VGGperceptualoss
    loss = VGGPerceptualLoss()
    a = torch.ones(10,3,224,224)
    b = torch.zeros(10,3,224,224)
    l = loss(a,b)
    print(l.data)