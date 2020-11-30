from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, f_q, f_k):
        # batch size, channel size, and number of sample locations
        B, C, H, W = f_q.shape
        S = H * W
        f_q = f_q.reshape(B, C, S)
        f_k = f_k.reshape(B, C, S)
        f_k = f_k.detach()

        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]

        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)

        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S, dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))

        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / self.tau

        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)


if __name__ == '__main__':
    q = torch.randn((2, 512, 16, 16))
    k = torch.randn((2, 512, 16, 16))

    loss = PatchNCELoss(0.05)

    out = loss(q, k)
