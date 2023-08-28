import torch 
from torch import nn
from Depthwise_Conv2D import depthwise_separable_conv
from SqExBlock import ResBlockSqEx
class MBConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MBConvBlock, self).__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(in_channel,out_channel, (1,1),stride=1, padding=0),
            depthwise_separable_conv(out_channel,out_channel,5),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            ResBlockSqEx(out_channel),
            nn.Conv2d(out_channel, out_channel, (1,1), stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.2)
        )
        
    def forward(self, input):
        return self.Layer(input)
     
        