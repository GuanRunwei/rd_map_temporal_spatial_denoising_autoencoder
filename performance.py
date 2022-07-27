import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from utils.dataloader import get_dataloader
from nets.nano_sta_decoder import nano_sta
from torch.utils.tensorboard import SummaryWriter
from utils.callbacks import loss_save
import datetime
import time
import matplotlib.pyplot as plt
import random
import os
from thop import profile
from thop import clever_format


def get_fps(input, model, test_times=100):
    t1 = time.time()

    for _ in range(test_times):
        output = model(input)

    t2 = time.time()

    return ((t2 - t1) / test_times), 1 / ((t2 - t1) / test_times)






if __name__ == '__main__':
    device = 'cpu'
    input = torch.randn(3, 1, 1, 128, 64).to(device)


    nano_sta_model = nano_sta(encoder_in_channels=1, encoder_out_channels=128, decoder_out_channels=1).to(device)
    macs, params = profile(nano_sta_model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print("========== clever mode ==========")
    print("macs:", macs)
    print("params:", params)
    latency, fps = get_fps(input=input, model=nano_sta_model, test_times=200)
    print("FPS:", fps)
    print("latency:", latency * 1000, " ms")


