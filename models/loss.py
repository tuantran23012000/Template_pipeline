import torch.nn.functional as F
import torch.nn as nn
class NllLoss(nn.Module):
    def __init__(self,cfgs):
        super(NllLoss,self).__init__()
        self.cfgs = cfgs
    def forward(self,output,target):
        loss = F.nll_loss(output, target)
        return loss