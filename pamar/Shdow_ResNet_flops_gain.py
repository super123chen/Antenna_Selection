
from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from torch.optim import AdamW
#from uitls_8822_gain import computation_time
import math
from time import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
#import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
import torchvision.models as models
# import torch
from ptflops import get_model_complexity_info

class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, bottle_channels,out_channels, stride=1):
        super(ResidualBottleneck, self).__init__()
        width = out_channels // 16
        self.conv1 = nn.Conv2d(in_channels, bottle_channels, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(bottle_channels)
        self.conv2 = nn.Conv2d(bottle_channels, bottle_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(bottle_channels)
        self.conv3 = nn.Conv2d(bottle_channels, out_channels, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.inc=in_channels
        self.outc=out_channels
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
    

        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.inc!=self.outc:
          residual = self.downsample(x)
        
       
        out += residual
        out = self.relu(out)
        return out









class Shadow(nn.Module):
    def __init__(self, inc):
        super(Shadow, self).__init__()
        self.lin1 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin2 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin3 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin4 = nn.Linear(int(inc / 4), int(inc / 4))
        self.conv = nn.Conv2d(int(2 * inc), inc, 1)
        

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # bhwc
        x_chunks = torch.chunk(x, chunks=4, dim=3)
        x_chunk1, x_chunk2, x_chunk3, x_chunk4 = x_chunks

        x_chunk1 = self.lin1(x_chunk1)
        x_chunk1 = F.relu(x_chunk1)
        x_chunk2 = self.lin2(x_chunk2)
        x_chunk2 = F.gelu(x_chunk2)
        x_chunk3 = self.lin3(x_chunk3)
        x_chunk3 = F.selu(x_chunk3)
        x_chunk4 = self.lin4(x_chunk4)
        x_chunk4 = x_chunk4*F.sigmoid(x_chunk4)
        

        x = torch.cat([x, x_chunk1, x_chunk2, x_chunk3, x_chunk4], dim=3)
        
       

        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)

       
        return x   




      
      
        
        
        








class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.convseq=nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=2, bias=True),
        Shadow(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, bias=True),
        Shadow(128),
        nn.ReLU6(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, bias=True),
        Shadow(256),
        nn.SELU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=2, bias=True),
        Shadow(256),
        nn.GELU(),
        nn.Conv2d(256, 128, kernel_size=2, bias=True),
        Shadow(128),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(128, 64, kernel_size=2,bias=True),
        Shadow(64),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
      
        
        )
        
        self.resblocks = nn.Sequential(
            ResidualBottleneck(64,16 ,64),         
            ResidualBottleneck(64, 32,128),            
            ResidualBottleneck(128, 32,128),         
            ResidualBottleneck(128, 64,256),            
            ResidualBottleneck(256, 64,256),
            
        )
      
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(256)
      
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)# wojiade         
        self.linear = nn.Linear(256, 784)
        

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # bchw
        x=self.convseq(x)
        
        
        x = self.resblocks(x)
        x = self.relu(self.bn(x))
        x = self.avg_pool(x)
        
        x_flattened = x.flatten(start_dim=1)  # bchw
        x = self.dropout(x)#  wojiade

        x = self.linear(x_flattened)
        return x

model = MyNetwork()


model_name = 'xxxx'
flops, params = get_model_complexity_info(model, (8, 8, 1), as_strings=True, print_per_layer_stat=True)
# print("%s |%s |%s" % (model_name, flops, params))
print("model_name",model_name)
print("flops:",flops)
print("params:",params)
