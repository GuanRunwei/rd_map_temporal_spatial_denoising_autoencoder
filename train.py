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
from utils.callbacks import loss_save
import datetime
import random
import os




if __name__ == '__main__':

    # -------------------------------- 超参数 -------------------------------- #
    data_path = "E:/Big_Datasets/RaDICaL_Denoising/RD_map_log/RD_map_log/temporal_spatial_data.pkl"
    batch_size = 16
    train_ratio = 0.92
    cuda = True
    optimizer_name = 'adam'
    scheduler_name = 'step'
    learning_rate = 0.001
    weight_decay = 5e-4
    epochs = 100
    criterion = nn.MSELoss()
    # ------------------------------------------------------------------------ #

    # ------------------------------- 训练设备 --------------------------------- #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -------------------------------------------------------------------------- #

    # --------------------------------- SEED ------------------------------------- #
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed(777)
    # ---------------------------------------------------------------------------- #

    # ================================================== 开始训练 ==================================================#

    # ------------------------------- 模型定义 --------------------------------- #
    nano_sta_model = nano_sta(encoder_in_channels=1, encoder_out_channels=128, decoder_out_channels=1).to(device)
    nano_sta_model._initialize_weights()
    # -------------------------------------------------------------------------- #

    # ------------------------------- 数据集加载 ------------------------------- #
    trainloader, testloader, validloader = get_dataloader(data_path=data_path, batch_size=batch_size,
                                                          train_ratio=train_ratio)
    # -------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------- #
    # writer = SummaryWriter("logs")
    # time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    # log_dir = os.path.join("logs", "loss_" + str(time_str) + '.txt')
    # log_history = open(log_dir, encoding='utf8', mode='w')
    # -------------------------------------------------------------------------- #

    # ------------------------------ Optimizer --------------------------------- #
    if optimizer_name == 'adam':
        optimizer = optim.AdamW(lr=learning_rate, params=nano_sta_model.parameters(), weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(lr=learning_rate, params=nano_sta_model.parameters(), momentum=0.937)
    # -------------------------------------------------------------------------- #

    # ------------------------------ Scheduler --------------------------------- #
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=learning_rate * 0.01, T_max=epochs/10)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.9, step_size=1)
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
    mse_loss_min = 1000000
    train_loss_array = []
    valid_loss_array = []
    best_model = None
    best_model_name = None
    for epoch in range(epochs):
        train_loss = 0
        train_loop = tqdm(enumerate(trainloader), total=len(trainloader))
        nano_sta_model.train()
        for i, (maps, labels) in train_loop:
            inputs = maps.to(device)
            gts = labels.to(device)
            predictions = nano_sta_model(inputs).to(device)
            loss = criterion(predictions, gts)
            train_loss += loss.item()

            # ------------------ 清空梯度,反向传播 ----------------- #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ----------------------------------------------------- #
            train_loop.set_description(f'Epoch [{epoch}/{epochs}]')
            train_loop.set_postfix(MSE_Loss=loss.item(), learning_rate=optimizer.param_groups[0]['lr'])
        train_loss_array.append(train_loss)

        # log_history.write(str(loss.item()) + '\n')

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
                validation_loss += loss.item()
                validation_loop.set_postfix(loss_real_time=loss.item(), Validation_MSE_Loss_Per_Map=
                                            validation_loss / (len(validloader) * batch_size))
            if validation_loss < mse_loss_min:
                best_model = nano_sta_model
                best_model_name = "val_loss_" + str(validation_loss) + '.pth'
                print("best model now:", best_model_name)
                torch.save(best_model.state_dict(), best_model_name)
                mse_loss_min = validation_loss
            # writer.add_scalar("MSE - Validation", validation_loss / (len(validloader) * batch_size), epoch)
            # if validation_loss / (len(validloader) * batch_size) <= mse_loss_min:
            #     torch.save(nano_sta_model.state_dict(), "logs/validation " +
            #                str(validation_loss / (len(validloader) * batch_size)) + ".pth")
        valid_loss_array.append(validation_loss)

        print()
        print("########################## end validation #############################")
        # ---------------------------------------------------------------------------- #

        scheduler.step()

    loss_save(train_loss_array, mode='train')
    loss_save(valid_loss_array, mode='valid', model=best_model, model_name=best_model_name)
    print()
    print("============================== end training =================================")











