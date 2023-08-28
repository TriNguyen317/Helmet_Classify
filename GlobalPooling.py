import torch
from torch import nn

class GlobalAvgPooling(nn.Module):
    def __init__(self) :
        super(GlobalAvgPooling,self).__init__()
    
    def forward(self, input, dim=0):
        output = torch.mean(input,dim)
        return torch.unsqueeze(output,1)

class GlobalMaxPooling(nn.Module):
    def __init__(self) -> None:
        super(GlobalMaxPooling,self).__init__()
        
    def forward(self, input, dim=0):
        _, output = torch.max(input, dim)
        return torch.unsqueeze(output,1)