import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from utils.dataloader import get_dataloader
from nets.mobile_encoder import mobile_encoder



if __name__ == '__main__':

    # -------------------------------- 超参数 -------------------------------- #
    data_path = "E:/Big_Datasets/RaDICaL_Denoising/temporal_spatial_data.pkl"
    batch_size = 8
    train_ratio = 0.9
    cuda = True
    optimizer_name = 'adam'
    scheduler_name = 'cosine'
    learning_rate = 0.0001
    weight_decay = 5e-4
    epochs = 50
    criterion = nn.MSELoss()
    # ------------------------------------------------------------------------ #


    # ------------------------------- 数据集加载 ------------------------------- #
    trainloader, testloader, validloader = get_dataloader(data_path=data_path, batch_size=batch_size,
                                                          train_ratio=train_ratio)
    # -------------------------------------------------------------------------- #

    device = 'cuda:0' if torch.cuda.is_available() and cuda == True else 'cpu'

    # if optimizer_name == 'adam':
    #     optimizer = optim.Adam(lr=learning_rate, weight_decay=5e-4)
    # elif optimizer_name == 'sgd':
    #     optimizer = optim.SGD(lr=learning_rate, momentum=0.937)

    maps, labels = next(iter(trainloader))
    print(maps.shape)

    model = mobile_encoder(in_channels=1, out_channels=128)
    gt_test = torch.randn((8, 128, 16, 8))
    output = model(maps)
    print("output shape:", output.shape)
    print("gt shape:", labels.shape)
    print("loss:", criterion(output, gt_test))


