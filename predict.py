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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import os

def plot_denoised_rd_maps(denoised_rd_maps):
    fig, ax = plt.subplots(3, 7, figsize=(15, 12))
    row_index = 0
    data_index = 1

    for index in range(denoised_rd_maps.shape[0]):
        rd_map_noise = denoised_rd_maps[index]
        ax[row_index, index % 7].imshow(rd_map_noise)
        ax[row_index, index % 7].set_title("Frame {},Noise {}".format(row_index + data_index, index % 7))

        if (index + 1) % 7 == 0 and index != 0:
            row_index += 1

        if row_index == 3:
            break

    plt.show()



def export_rd_map():
    model_path = "models/nano_sta.pth"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = nano_sta(encoder_in_channels=1, encoder_out_channels=128, decoder_out_channels=1).to('cpu')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    final_results = []
    gts = []
    mapts = []

    train_data_read = pd.read_pickle(
        "E:/Big_Datasets/RaDICaL_Denoising/RD_map_log/RD_map_log/temporal_spatial_data.pkl")
    print("======== start export ==========")
    for i in trange(14, 35):
        example_data = train_data_read.loc[i]
        map_t, map_t_1, map_t_2, gt = example_data['t'], example_data['t-1'], example_data['t-2'], \
                                      example_data['gt']

        input_map = torch.from_numpy(np.array(map_t)).unsqueeze(1).unsqueeze(2).type(
            torch.FloatTensor)

        output_map = model(input_map).squeeze(0).squeeze(0)
        final_results.append(output_map.detach().numpy())
        gts.append(gt)
        mapts.append(map_t)

    final_results = np.array(final_results)
    final_results = final_results.reshape(final_results.shape[0], final_results.shape[1]*final_results.shape[2], -1)
    gts = np.array(gts)
    mapts = np.array(mapts)
    print(final_results.shape)
    print(gts.shape)
    print(mapts.shape)
    np.save("cae.npy", final_results)
    np.save("gt_maps.npy", gts)
    np.save("mapts.npy", mapts)

    plot_denoised_rd_maps(final_results)
    print("======== end export ==========")





if __name__ == '__main__':

    model_path = "models/nano_sta.pth"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    nano_sta_model = nano_sta(encoder_in_channels=1, encoder_out_channels=128, decoder_out_channels=1).to(device)
    nano_sta_model.load_state_dict(torch.load(model_path))
    nano_sta_model.eval()

    train_data_read = pd.read_pickle(
        "E:/Big_Datasets/RaDICaL_Denoising/RD_map_log/RD_map_log/temporal_spatial_data.pkl")
    random_index = random.randint(20000, 23966)
    print("random index:", 0)
    example_data = train_data_read.loc[random_index]
    map_t, map_t_1, map_t_2, gt = example_data['t'], example_data['t-1'], example_data['t-2'], \
                                  example_data['gt']

    np.save("qualitive_rd_map_example.npy", map_t)

    input_map = torch.from_numpy(np.array([map_t, map_t_1, map_t_2])).unsqueeze(1).unsqueeze(2).type(torch.cuda.FloatTensor)
    print(input_map.shape)

    output_map = nano_sta_model(input_map)

    output_map = output_map.squeeze(0).squeeze(0)
    print(output_map.shape)
    output_map = output_map.detach().to('cpu').numpy()


    fig, ax = plt.subplots(1, 5, figsize=(15, 7))
    ax[0].imshow(map_t)
    ax[0].set_title("Frame t interference")

    ax[1].imshow(map_t_1)
    ax[1].set_title("Frame t-1 interference")

    ax[2].imshow(map_t_2)
    ax[2].set_title("Frame t-2 interference")

    ax[3].imshow(gt)
    ax[3].set_title("Frame t gt")

    ax[4].imshow(output_map)
    ax[4].set_title("Frame t decouple")

    plt.imshow(output_map)
    now = datetime.datetime.now()
    plt.savefig(f"images/test_{random_index}_{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}.png")
    plt.show()

    export_rd_map()

