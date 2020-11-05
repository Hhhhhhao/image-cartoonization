import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

__all__ = ['BCEGANLoss', 'LSGANLoss', 'WGANGPLoss']


class BCEGANLoss(nn.Module):
    def __init__(self):
        super(BCEGANLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.register_buffer('target_real', torch.tensor(1.0))
        self.register_buffer('target_fake', torch.tensor(0.0))

    def get_target_tensor(self, pred_logits, real=True):
        if real is True:
            target_tensor = self.target_real
        else:
            target_tensor = self.target_fake
        return target_tensor.expand_as(pred_logits).to(pred_logits.device)

    def forward(self, pred_logits, real=True):
        pred = F.sigmoid(pred_logits)
        return self.bce_loss(pred, self.get_target_tensor(pred_logits, real))


class LSGANLoss(nn.Module):
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.register_buffer('target_real', torch.tensor(1.0))
        self.register_buffer('target_fake', torch.tensor(0.0))

    def get_target_tensor(self, pred_logits, real=True):
        if real is True:
            target_tensor = self.target_real
        else:
            target_tensor = self.target_fake
        return target_tensor.expand_as(pred_logits).to(pred_logits.device)

    def forward(self, pred_logits, real=True):
        return self.mse_loss(pred_logits, self.get_target_tensor(pred_logits, real))


class WGANGPLoss(nn.Module):
    def __init__(self, lambda_=10):
        super(WGANGPLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, pred_logits, real=True):

        if real:
            return -torch.mean(pred_logits)
        else:
            return torch.mean(pred_logits)

    def gradient_penalty(self, real_x, fake_x, discriminator):
        alpha = torch.rand((self.batch_size, 1, 1, 1)).to(real_x.device)
        x_hat = alpha * real_x.data + (1 - alpha) * fake_x.data
        x_hat.requires_grad = True

        pred_hat = discriminator(x_hat)

        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(real_x.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        return gradient_penalty