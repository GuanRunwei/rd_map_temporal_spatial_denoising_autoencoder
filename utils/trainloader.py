import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os



class RD_Trainset(Dataset):
    def __init__(self, trainset_path, transforms=None):
        self.trainset_path = trainset_path
        self.transforms = transforms

    # 预处理npy文件格式，将sb0和sb的维度从64，128，10272 -> 10272，128，64
    def _preprocess_map_format(self):
        trainset = np.load(self.trainset_path, allow_pickle=True)
        trainset = torch.as_tensor(trainset).permute(2, 1, 0)
        return trainset

    def __getitem__(self, index):
        trainset = self._preprocess_map_format()
        