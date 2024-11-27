import argparse
import datetime

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import dataloader
import random
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
import numpy as np
import sys
import time
import shutil
import errno
from models.ACGAN import Generator, Discriminator
from dataset.dataset2 import dataset_for_image, dataset_for_radar, dataset_for_ADSB

device = torch.device('cuda:0')


# model = Generator(in_channels=2, num_class=4)



backbone_weight_path = './log/ACGAN/ADSB/202411072316/result_model/_g_last.pth'
fake_data_save_folder = './log/ACGAN/ADSB/202411072316/result_model/fakedata/'
model = torch.load(backbone_weight_path)
model = model.to(device)
# model.load_state_dict(torch.load(backbone_weight_path))
for i in range(15):
    noise = torch.randn(int(50), 100).cuda()
    fake_label = i*torch.ones(int(50), dtype=torch.int).cuda()
    # model.eval()
    fakedata = model(noise, fake_label).detach()
    fakedata = np.array(fakedata.cpu().numpy())
    if not os.path.exists(fake_data_save_folder):
        os.makedirs(fake_data_save_folder)
    fakedata_save_path = fake_data_save_folder +str(i)+'.npy'
    np.save(fakedata_save_path, fakedata)

