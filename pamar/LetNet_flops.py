
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







class CNN(nn.Module):
    def __init__(self,n_class):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=784)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)
       
        x = self.features(x)
        
        x = torch.flatten(x, 1)
        
        result = self.classifier(x)
        return result


n_class = 784  
model = CNN(n_class)












model_name = 'xxxx'
flops, params = get_model_complexity_info(model, (8, 8, 1), as_strings=True, print_per_layer_stat=True)
# print("%s |%s |%s" % (model_name, flops, params))
print("model_name",model_name)
print("flops:",flops)
print("params:",params)
