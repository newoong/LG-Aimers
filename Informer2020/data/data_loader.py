import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

# from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, data_x, data_y, x_mark, y_mark):
        self.data_x = data_x
        self.data_y = data_y
        self.x_mark = x_mark
        self.y_mark = y_mark
    
    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark = self.x_mark[index]
        seq_y_mark = self.y_mark[index]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x)
