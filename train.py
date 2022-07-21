import numpy as np
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
from utils.callbacks import LossHistory
import datetime
import os




if __name__ == '__main__':

    # -------------------------------- 超参数 -------------------------------- #
    data_path = "E:/Big_Datasets/RaDICaL_Denoising/temporal_spatial_data.pkl"
    batch_size = 8
    train_ratio = 0.9
    cuda = True
    optimizer_name = 'sgd'
    scheduler_name = 'cosine'
    learning_rate = 0.0003
    weight_decay = 5e-4
    epochs = 50
    criterion = nn.MSELoss()
    # ------------------------------------------------------------------------ #

    # ------------------------------- 训练设备 --------------------------------- #
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # -------------------------------------------------------------------------- #

    # ================================================== 开始训练 ==================================================#

    # ------------------------------- 模型定义 --------------------------------- #
    nano_sta_model = nano_sta(encoder_in_channels=1, encoder_out_channels=256, decoder_out_channels=1).to(device)
    # -------------------------------------------------------------------------- #

    # ------------------------------- 数据集加载 ------------------------------- #
    trainloader, testloader, validloader = get_dataloader(data_path=data_path, batch_size=batch_size,
                                                          train_ratio=train_ratio)
    # -------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------- #
    writer = SummaryWriter("logs")
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join("logs", "loss_" + str(time_str) + '.txt')
    log_history = open(log_dir, encoding='utf8', mode='w')
    # -------------------------------------------------------------------------- #

    # ------------------------------ Optimizer --------------------------------- #
    if optimizer_name == 'adam':
        optimizer = optim.Adam(lr=learning_rate, params=nano_sta_model.parameters(), weight_decay=5e-4)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(lr=learning_rate, params=nano_sta_model.parameters(), momentum=0.937)
    # -------------------------------------------------------------------------- #

    # ------------------------------ Scheduler --------------------------------- #
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=0.005 * 0.05, T_max=epochs/10)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.1)
    # -------------------------------------------------------------------------- #

    # ------------------------------ Start Training ---------------------------- #
    print()
    print("================= Training Configuration ===================")
    print("trainloader size:", len(trainloader) * batch_size)
    print("validloader size:", len(validloader) * batch_size)
    print("testloader size:", len(testloader) * batch_size)
    print("epoch:", epochs)
    print("batch size:", batch_size)
    print("optimizer:", optimizer_name)
    print("scheduler:", scheduler_name)
    print("initial learning rate:", learning_rate)
    print("weight decay:", weight_decay)
    print("=============================================================")
    mse_loss_min = 10000
    for epoch in range(epochs):
        train_loss = 0
        train_loop = tqdm(enumerate(trainloader), total=len(trainloader))
        nano_sta_model.train()
        for i, (maps, labels) in train_loop:
            inputs = maps.to(device)
            gts = labels.to(device)
            predictions = nano_sta_model(inputs).to(device)
            loss = criterion(predictions, gts)
            train_loss += loss

            # ------------------ 清空梯度,反向传播 ----------------- #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ----------------------------------------------------- #
            train_loop.set_description(f'Epoch [{epoch}/{epochs}]')
            train_loop.set_postfix(MSE_Loss=loss.item(), learning_rate=optimizer.param_groups[0]['lr'])

        log_history.write(str(loss.item()) + '\n')

        # loss_history = LossHistory(log_dir="logs", model=nano_sta_model, input_shape=[128, 64])

        # ------------------------------- Validation --------------------------------- #
        validation_loop = tqdm(enumerate(validloader), total=len(validloader))

        print()
        print("########################## start validation #############################")
        nano_sta_model.eval()
        with torch.no_grad():
            validation_loss = 0
            for i, (maps, labels) in validation_loop:
                inputs = maps.to(device)
                gts = labels.to(device)
                valid_prediction = nano_sta_model(inputs)
                loss = criterion(valid_prediction, gts)
                validation_loss += loss
                validation_loop.set_postfix(Validation_MSE_Loss_Per_Map=
                                            validation_loss / (len(validloader) * batch_size))
            writer.add_scalar("MSE - Validation", validation_loss / (len(validloader) * batch_size), epoch)
            if validation_loss / (len(validloader) * batch_size) <= mse_loss_min:
                torch.save(nano_sta_model.state_dict(), "logs/validation " +
                           str(validation_loss / (len(validloader) * batch_size)) + ".pth")

        print()
        print("########################## end validation #############################")
        # ---------------------------------------------------------------------------- #

        scheduler.step()

    print()
    print("============================== end training =================================")











