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
        # unclear about the loss in gan, skip for now
        raise NotImplementedError('Need to build criterion')

    def _build_metrics(self):
        self.metric_names = ['disc','gen']
        self.train_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)
        self.valid_metrics = MetricTracker(*[metric for metric in self.metric_names], writer=self.writer)


    def _train_epoch(self, epoch):

        self.gen.train()
        self.disc.train()
        self.train_metrics.reset()

        for batch_index,(src_imgs, tar_imgs) in enumerate(self.train_dataloader):



if __name__ == "__main__":
    # test VGGperceptualoss
    loss = VGGPerceptualLoss()
    a = torch.randn(1,3,224,224)
    b = torch.randn(1,3,224,224)
    l = loss(a,b)
    print(l.data)