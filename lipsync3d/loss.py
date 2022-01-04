import torch
import torch.nn as nn

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, input, target):
        l2_loss = (target - input) ** 2
        l2_loss = torch.mean(l2_loss)

        return l2_loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, input, target):
        
        return (torch.abs(input - target)).mean()