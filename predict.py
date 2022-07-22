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
import matplotlib.pyplot as plt
import random
import os


if __name__ == '__main__':

    model_path = "models/val_loss_0.22674559114966542.pth"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    nano_sta_model = nano_sta(encoder_in_channels=1, encoder_out_channels=128, decoder_out_channels=1).to(device)
    nano_sta_model.load_state_dict(torch.load(model_path))
    nano_sta_model.eval()

    train_data_read = pd.read_pickle(
        "E:/Big_Datasets/RaDICaL_Denoising/RD_map_log/RD_map_log/temporal_spatial_data.pkl")
    random_index = random.randint(20000, 23966)
    print("random index:", random_index)
    example_data = train_data_read.loc[random_index]
    map_t, map_t_1, map_t_2, gt = example_data['t'], example_data['t-1'], example_data['t-2'], \
                                  example_data['gt']

    input_map = torch.from_numpy(np.array([map_t, map_t_1, map_t_2])).unsqueeze(1).unsqueeze(2).type(torch.cuda.FloatTensor)
    print(input_map.shape)

    output_map = nano_sta_model(input_map)
    output_map = output_map.squeeze(0).squeeze(0)
    output_map = output_map.detach().to('cpu').numpy()

    fig, ax = plt.subplots(1, 5, figsize=(15, 7))
    ax[0].imshow(map_t)
    ax[0].set_title("Frame T interference")

    ax[1].imshow(map_t_1)
    ax[1].set_title("Frame T-1 interference")

    ax[2].imshow(map_t_2)
    ax[2].set_title("Frame T-2 interference")

    ax[3].imshow(gt)
    ax[3].set_title("Frame T gt")

    ax[4].imshow(output_map)
    ax[4].set_title("Frame T decouple")

    plt.imshow(output_map)
    now = datetime.datetime.now()
    plt.savefig(f"images/test{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}.png")


