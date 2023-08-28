import torch 
from torch import nn

class Channel_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Channel_Attention,self).__init__()
        self.AvgPool = nn.AvgPool2d(46)
        self.fc_1 = nn.Linear(in_channel,32)
        self.fc_2 = nn.Linear(32,in_channel)
        
        self.MaxPool = nn.MaxPool2d(46)
        self.fc_3 = nn.Linear(in_channel,32)
        self.fc_4 = nn.Linear(32,in_channel)
        
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self,input):
        AvgPool = self.AvgPool(input)
        shape = AvgPool.shape
        AvgPool = torch.flatten(AvgPool, start_dim=1)
        AvgPool = self.fc_1(AvgPool)
        AvgPool = self.fc_2(AvgPool)
        AvgPool = torch.reshape(AvgPool, shape)
        
        MaxPool = self.MaxPool(input)
        MaxPool = torch.flatten(MaxPool, start_dim=1)
        MaxPool = self.fc_3(MaxPool)
        MaxPool = self.fc_4(MaxPool)
        MaxPool = torch.reshape(MaxPool, shape)
        
        cbam_feature = MaxPool + AvgPool
        cbam_feature = self.softmax(cbam_feature)
        
        output= cbam_feature * input
        
        return output