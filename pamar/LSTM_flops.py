


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















device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
      #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      x = x.view(x.size(0), -1)
      h0 = torch.zeros(2, x.size(0), self.lstm.hidden_size).to(x.device)
      c0 = torch.zeros(2, x.size(0), self.lstm.hidden_size).to(x.device)
      #out, _ = self.lstm(x, (h0, c0))
      out, _ = self.lstm(x.unsqueeze(1), (h0,c0))
      
       
      output = self.fc(out.squeeze(1))
      return output
      



      



model = LSTM(input_size=64, hidden_size=256, num_layers=2, output_size=784)
x = torch.randn(16, 1, 64).to(device)  




model_name = 'xxxx'
flops, params = get_model_complexity_info(model, (8, 8, 1), as_strings=True, print_per_layer_stat=True)
# print("%s |%s |%s" % (model_name, flops, params))
print("model_name",model_name)
print("flops:",flops)
print("params:",params)




