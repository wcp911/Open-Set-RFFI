import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, KMNIST

import numpy as np
from PIL import Image

import sys

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import scipy.io as scio


class dataset_for_image(Dataset):
    """
        mode_type == 'train':仅加载已知类所有文件
        mode_type == 'closeval':加载已知类所有文件
        mode_type == 'openval':加载所有类别所有文件
        mode_type == 'train':加载所有类别指定信噪比文件

        if known_typelist == None:
            known_typelist = range(0, self.class_num)

    """
    def __init__(self, data_root, dataset='MNIST', target_typelist='all', mode_type='train'):
        self.target_typelist = target_typelist
        self.dataset = dataset
        self.mode_type = mode_type
        self.train_data_root = os.path.join(data_root, 'train')
        self.val_data_root = os.path.join(data_root, 'val')
        self.test_data_root = os.path.join(data_root, 'test')
        self.typelist = os.listdir(self.train_data_root)  # 获取数据文件夹中的所有文件，即信号种类名称
        self.class_num = len(self.typelist)
        self.datas, self.target_name, self.target = [], [], []
        if self.target_typelist == 'all':
            self.target_typelist = range(0, self.class_num)

        if self.dataset == 'tiny_imagenet':
            image_size = 64
        else:
            image_size = 32
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        if mode_type == 'train':
            for i in range(len(self.target_typelist)):
                target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
                if self.dataset == 'tiny_imagenet':
                    image_list = os.listdir(os.path.join(self.train_data_root, target_name, 'images'))  # 获取数据文件夹中的所有mat名称
                else:
                    image_list = os.listdir(os.path.join(self.train_data_root, target_name))  # 获取数据文件夹中的所有mat名称
                image_num = len(image_list)  # 该类mat文件的总数量
                for j in range(image_num):
                    if self.dataset == 'tiny_imagenet':
                        data_value = Image.open(os.path.join(self.train_data_root, target_name, 'images', image_list[j])).convert('RGB')
                    else:
                        data_value = Image.open(os.path.join(self.train_data_root, target_name, image_list[j])).convert('RGB')
                    target = i  # 类别的序号（int）
                    self.datas.append(data_value)
                    self.target_name.append(target_name)
                    self.target.append(target)
                    # data_value.close()

        elif mode_type == 'test':
            for i in range(len(self.target_typelist)):
                target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
                image_list = os.listdir(os.path.join(self.test_data_root, target_name))  # 获取数据文件夹中的所有mat名称
                image_num = len(image_list)  # 该类mat文件的总数量
                for j in range(image_num):
                    data_value = Image.open(os.path.join(self.test_data_root, target_name, image_list[j])).convert('RGB')
                    target = i  # 类别的序号（int）
                    self.datas.append(data_value)
                    self.target_name.append(target_name)
                    self.target.append(target)
                    # data_value.close()

    def __getitem__(self, index):
        data = self.datas[index]
        target_name = self.target_name[index]
        target = self.target[index]
        if self.mode_type == 'train':
            data = self.train_transform(data)
        else:
            data = self.test_transform(data)
        return data, target_name, target

    def __len__(self):
        return len(self.datas)

class dataset_for_radar(Dataset):
    """
        mode_type == 'train':仅加载已知类所有文件
        mode_type == 'closeval':加载已知类所有文件
        mode_type == 'openval':加载所有类别所有文件
        mode_type == 'train':加载所有类别指定信噪比文件

        if known_typelist == None:
            known_typelist = range(0, self.class_num)

    """

    def __init__(self, data_root, target_typelist='all', mode_type='train', SNR=None):
        self.target_typelist = target_typelist
        self.train_data_root = os.path.join(data_root, 'train')
        self.val_data_root = os.path.join(data_root, 'val')
        self.test_data_root = os.path.join(data_root, 'test')
        self.typelist = os.listdir(self.train_data_root)  # 获取数据文件夹中的所有文件，即信号种类名称
        self.class_num = len(self.typelist)
        self.datas, self.target_name, self.target = [], [], []
        if self.target_typelist == 'all':
            self.target_typelist = range(0, self.class_num)

        if mode_type == 'train':
            for i in range(len(self.target_typelist)):
                target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
                snr_list = os.listdir(os.path.join(self.train_data_root, target_name))  # 获取所有的信噪比文件夹
                for snr in snr_list:
                    mat_list = os.listdir(os.path.join(self.train_data_root, target_name, snr))  # 获取数据文件夹中的所有mat名称
                    mat_num = len(mat_list)  # 该类mat文件的总数量
                    for j in range(mat_num):
                        data_temp = scio.loadmat(
                            os.path.join(self.train_data_root, target_name, snr, mat_list[j]))  # 加载mat文件
                        # 'real_UpEdgeSig','imag_UpEdgeSig','real_DownEdgeSig','imag_DownEdgeSig'
                        data_value = np.concatenate((data_temp['real_UpEdgeSig'], data_temp['imag_UpEdgeSig']),
                                                    axis=1)
                        data_value = data_value.reshape(2, 1024)
                        target = i  # 类别的序号（int）
                        self.datas.append(data_value)
                        self.target_name.append(target_name)
                        self.target.append(target)

        elif mode_type == 'val':
            for i in range(len(self.target_typelist)):
                target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
                snr_list = os.listdir(os.path.join(self.val_data_root, target_name))  # 获取所有的信噪比文件夹
                for snr in snr_list:
                    mat_list = os.listdir(os.path.join(self.val_data_root, target_name, snr))  # 获取数据文件夹中的所有mat名称
                    mat_num = len(mat_list)  # 该类mat文件的总数量
                    for j in range(mat_num):
                        data_temp = scio.loadmat(
                            os.path.join(self.val_data_root, target_name, snr, mat_list[j]))  # 加载mat文件
                        data_value = np.concatenate((data_temp['real_UpEdgeSig'], data_temp['imag_UpEdgeSig']),
                                                    axis=1)
                        data_value = data_value.reshape(2, 1024)
                        target = i  # 类别的序号（int）
                        self.datas.append(data_value)
                        self.target_name.append(target_name)
                        self.target.append(target)

        elif mode_type == 'test':
            for i in range(len(self.target_typelist)):
                target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
                snr_list = os.listdir(os.path.join(self.test_data_root, target_name))  # 获取所有的信噪比文件夹
                for snr in snr_list:
                    mat_list = os.listdir(os.path.join(self.test_data_root, target_name, snr))  # 获取数据文件夹中的所有mat名称
                    mat_num = len(mat_list)  # 该类mat文件的总数量
                    for j in range(mat_num):
                        data_temp = scio.loadmat(
                            os.path.join(self.test_data_root, target_name, snr, mat_list[j]))  # 加载mat文件
                        data_value = np.concatenate((data_temp['real_UpEdgeSig'], data_temp['imag_UpEdgeSig']),
                                                    axis=1)
                        data_value = data_value.reshape(2, 1024)
                        target = i  # 类别的序号（int）
                        self.datas.append(data_value)
                        self.target_name.append(target_name)
                        self.target.append(target)
        # elif mode_type == 'test':
        #     for i in range(len(self.target_typelist)):
        #         target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
        #         if SNR == None:
        #             print("错误：snr 为 None，请提供有效数值")
        #             sys.exit(1)  # 1 表示程序异常退出
        #         else:
        #             snr = str(SNR)
        #
        #         mat_list = os.listdir(os.path.join(self.test_data_root, target_name, snr))  # 获取数据文件夹中的所有mat名称
        #         mat_num = len(mat_list)  # 该类mat文件的总数量
        #         for j in range(mat_num):
        #             data_temp = scio.loadmat(
        #                 os.path.join(self.test_data_root, target_name, snr, mat_list[j]))  # 加载mat文件
        #             data_value = np.concatenate((data_temp['real_UpEdgeSig'], data_temp['imag_UpEdgeSig']), axis=1)
        #             data_value = data_value.reshape(2, 1024)
        #             target = i  # 类别的序号（int）
        #             self.datas.append(data_value)
        #             self.target_name.append(target_name)
        #             self.target.append(target)

    def __getitem__(self, index):
        data = self.datas[index]
        target_name = self.target_name[index]
        target = self.target[index]
        data = torch.tensor(data).type(torch.FloatTensor)  # 将数据转换为tensor
        return data, target_name, target

    def __len__(self):
        return len(self.datas)


class dataset_for_ADSB(Dataset):
    """
        mode_type == 'train':仅加载已知类所有文件
        mode_type == 'closeval':加载已知类所有文件
        mode_type == 'openval':加载所有类别所有文件
        mode_type == 'train':加载所有类别指定信噪比文件

        if known_typelist == None:
            known_typelist = range(0, self.class_num)

    """

    def __init__(self, data_root, target_typelist='all', mode_type='train'):
        self.target_typelist = target_typelist
        self.train_data_root = os.path.join(data_root, 'train')
        self.val_data_root = os.path.join(data_root, 'val')
        self.test_data_root = os.path.join(data_root, 'test')
        self.typelist = os.listdir(self.train_data_root)  # 获取数据文件夹中的所有文件，即信号种类名称
        self.class_num = len(self.typelist)
        self.datas, self.target_name, self.target = [], [], []
        if self.target_typelist == 'all':
            self.target_typelist = range(0, self.class_num)

        if mode_type == 'train':
            for i in range(len(self.target_typelist)):
                target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
                mat_list = os.listdir(os.path.join(self.train_data_root, target_name))  # 获取数据文件夹中的所有mat名称
                mat_num = len(mat_list)  # 该类mat文件的总数量
                for j in range(mat_num):
                    data_temp = scio.loadmat(os.path.join(self.train_data_root, target_name, mat_list[j]))  # 加载mat文件
                    data_value = data_temp['data']
                    target = i  # 类别的序号（int）
                    self.datas.append(data_value)
                    self.target_name.append(target_name)
                    self.target.append(target)

        elif mode_type == 'val':
            for i in range(len(self.target_typelist)):
                target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
                mat_list = os.listdir(os.path.join(self.val_data_root, target_name))  # 获取数据文件夹中的所有mat名称
                mat_num = len(mat_list)  # 该类mat文件的总数量
                for j in range(mat_num):
                    data_temp = scio.loadmat(os.path.join(self.val_data_root, target_name, mat_list[j]))  # 加载mat文件
                    data_value = data_temp['data']
                    target = i  # 类别的序号（int）
                    self.datas.append(data_value)
                    self.target_name.append(target_name)
                    self.target.append(target)

        elif mode_type == 'test':
            for i in range(len(self.target_typelist)):
                target_name = self.typelist[self.target_typelist[i]]  # 类别的名称（str）
                mat_list = os.listdir(os.path.join(self.test_data_root, target_name))  # 获取数据文件夹中的所有mat名称
                mat_num = len(mat_list)  # 该类mat文件的总数量
                for j in range(mat_num):
                    data_temp = scio.loadmat(os.path.join(self.test_data_root, target_name, mat_list[j]))  # 加载mat文件
                    data_value = data_temp['data']
                    target = i  # 类别的序号（int）
                    self.datas.append(data_value)
                    self.target_name.append(target_name)
                    self.target.append(target)

    def __getitem__(self, index):
        data = self.datas[index]
        target_name = self.target_name[index]
        target = self.target[index]
        data = torch.tensor(data).type(torch.FloatTensor)  # 将数据转换为tensor
        return data, target_name, target

    def __len__(self):
        return len(self.datas)
