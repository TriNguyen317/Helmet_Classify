import torch
from torch import nn
from GlobalPooling import *
class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention, self).__init__()
        self.GMaxPool = GlobalAvgPooling()
        self.GAvgPool = GlobalAvgPooling()
        self.Conv2D_1 = nn.Conv2d(2,1,(7,7), stride=1, padding=3)
        self.softmax_1 = nn.Softmax()
    
    def forward(self,input):
        
        cbam_feature = torch.cat((self.GMaxPool(input,1),self.GAvgPool(input,1)),1)
        cbam_feature = self.Conv2D_1(cbam_feature)
        output = cbam_feature * input
        
        return output