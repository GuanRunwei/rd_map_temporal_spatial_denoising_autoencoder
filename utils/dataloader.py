import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os


class RD_Dataset(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.dataset_path = dataset_path
        self.transforms = transforms

        # 在__init__函数中一次加载和预处理整个数据集
        self.dataset = pd.read_pickle(dataset_path)

    def __len__(self):
        return len(self.dataset)

    # 预处理npy文件格式，将sb0和sb的维度从64，128，10272 -> 10272，128，64
    def _preprocess_map_format(self, data):
        dataset = torch.as_tensor(data).permute(2, 1, 0)
        return dataset

    def __getitem__(self, index):
        noise_map_t = self.dataset['t'][index]
        noise_map_t_1 = self.dataset['t-1'][index]
        noise_map_t_2 = self.dataset['t-2'][index]
        no_noise_map = self.dataset['gt'][index]

        return [noise_map_t, noise_map_t_1, noise_map_t_2], no_noise_map


def dataset_collate(batch):
    t_maps = []
    t_1_maps = []
    t_2_maps = []
    t_labels = []
    for maps, label in batch:
        t_maps.append(maps[0])
        t_1_maps.append(maps[1])
        t_2_maps.append(maps[2])
        t_labels.append(label)

    maps = torch.from_numpy(np.array([t_maps, t_1_maps, t_2_maps], dtype=np.float64)).unsqueeze(2).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(t_labels, dtype=np.float64)).unsqueeze(1).type(torch.FloatTensor)
    return maps, labels




def get_dataloader(data_path="E:/Big_Datasets/RaDICaL_Denoising/temporal_spatial_data.pkl", train_ratio=0.8,
                   batch_size=8):
    dataset = RD_Dataset(dataset_path="E:/Big_Datasets/RaDICaL_Denoising/temporal_spatial_data.pkl")

    # --------------- 设置训练、测试、验证数据索引 ----------------- #
    datalen = dataset.__len__()
    dataidx = np.array(list(range(datalen)))
    np.random.shuffle(dataidx)

    splitfrac = train_ratio
    split_idx = int(splitfrac * datalen)
    train_idxs = dataidx[:split_idx]
    valid_idxs = dataidx[split_idx]

    testsplit = 0.1
    testidxs = int(testsplit * len(train_idxs))

    test_idxs = train_idxs[:testidxs]
    train_idxs = train_idxs[testidxs:]

    np.random.shuffle(test_idxs)

    train_samples = torch.utils.data.SubsetRandomSampler(train_idxs)
    valid_samples = torch.utils.data.SubsetRandomSampler(valid_idxs)
    test_samples = torch.utils.data.SubsetRandomSampler(test_idxs)
    # ------------------------------------------------------------- #

    # --------------- 训练、测试、验证dataloader ----------------- #
    trainloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_samples, collate_fn=dataset_collate)
    testloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_samples, collate_fn=dataset_collate)
    validloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=valid_samples, collate_fn=dataset_collate)

    return trainloader, testloader, validloader


if __name__ == '__main__':
    dataset = RD_Dataset(dataset_path="E:/Big_Datasets/RaDICaL_Denoising/temporal_spatial_data.pkl")

    dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=dataset_collate)
    maps, label = next(iter(dataloader))

    print(maps[0].shape)
    print(len(maps))
    # print(maps[1][0])
    # print(maps[2][0])
