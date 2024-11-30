import importlib

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
from openpyxl import load_workbook
from sklearn import svm


import argparse
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import dataloader
import random
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import sys
import time
import shutil
import errno
from dataset.dataset2 import dataset_for_image, dataset_for_radar, dataset_for_ADSB
from models.resnet50_for_1d import ResNet50
from utils.openSVDD import openSVDD
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import KernelPCA


parser = argparse.ArgumentParser("Training")

# optimization
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model_name', type=str, default='ResNet50')

# misc
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(options):
    seed_everything(options['seed'])
    data_root_path = options['data_root'] + options['dataset']
    # Dataset
    if options['dataset'] == 'radar':
        trainset = dataset_for_radar(data_root_path, options['train_typelist'], mode_type='train')
    else:
        trainset = dataset_for_ADSB(data_root_path,  options['train_typelist'], mode_type='train')

    #  dataloader
    trainloader = DataLoader(trainset, batch_size=options['batch_size'], shuffle=True, pin_memory=True)
    print("All training data:", len(trainset))

    options['num_classes'] = len(options['train_typelist'])
    feat_dim = len(options['train_typelist'])
    # Model
    print('==> Building model: ' + options['model_name'])
    if options['dataset'] == 'radar':
        model = ResNet50(in_channels=2, feature_dim=feat_dim,
                         classes=len(options['train_typelist']))
    else:
        model = ResNet50(in_channels=3, feature_dim=len(options['train_typelist']),
                         classes=len(options['train_typelist']))
    # print(model)

    options.update(
        {
            'use_gpu': True,
            'feat_dim': feat_dim,
        }
    )

    Loss = importlib.import_module('loss.' + options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)
    model = nn.DataParallel(model).cuda()
    criterion = criterion.cuda()
    params = [{'params': model.parameters()},
              {'params': criterion.parameters()}]
    print(options['model_name'])
    loss_weight_path = options['loss_path']
    criterion.load_state_dict(torch.load(loss_weight_path))
    backbone_weight_path = options['weight_path']
    model.load_state_dict(torch.load(backbone_weight_path))
    train_label_list, train_outputs_list, test_label_list, test_outputs_list, test_label_list2, test_outputs_list2 = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for data in trainloader:
            inputs, target_name, target = data
            inputs = inputs.cuda()
            with torch.set_grad_enabled(False):  # 与上同理
                # ------------------------------------------------------------
                y = model(inputs)

            train_outputs_list.append(y.cpu().numpy())
            train_label_list.extend(target)
    train_outputs_list = np.concatenate(train_outputs_list, 0)
    train_label_list = np.array(train_label_list)

    labels_binary = [0 if x >= len(options['train_typelist']) else 1 for x in test_label_list2]

    C=1
    gamma = 1
    opensvdd = openSVDD(C=C, gamma=gamma, kernel='rbf')
    opensvdd.fit_openSVDD(train_outputs_list, train_label_list)
    test_label_list2[test_label_list2 >= len(options['train_typelist'])] = -1
    _, _, acc, _ = opensvdd.predict(train_outputs_list, train_label_list)
    print('trainset accuracy: %.4f' % (acc))
    # save model
    openSVDD_save_path = options['openSVDD_save_path'] + '/C' + str(C) + 'gamma' + str(gamma) + 'bestacc.pkl'
    joblib.dump(opensvdd, openSVDD_save_path)
    # opensvdd = joblib.load('./log/trainset_output_opensvdd_C0.9.pkl')


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    results = dict()

    from split import train_splits as train_splits
    from split import test_splits as test_splits
    model_name = 'ResNet50'

    test_list = ['radar', 'ADSB']
    for j in range(len(test_list)):
        torch.cuda.empty_cache()
        options.update(
            {
                'dataset': test_list[j]
            }
        )

        result_model_path = './log/ARPL/' + model_name + '/' + options['dataset']
        files = os.listdir(result_model_path)

        for i in range(len(train_splits[options['dataset']])):
            train_split = train_splits[options['dataset']][i]
            test_split = test_splits[options['dataset']][i]

            weight_path = result_model_path + '/' + files[i] + '/result_model/models/checkpoints/ResNet1D_ARPLoss_.pth'
            loss_path = result_model_path + '/' + files[i] + '/result_model/models/checkpoints/ResNet1D_ARPLoss__criterion.pth'
            openSVDD_save_path = result_model_path + '/' + files[i] + '/result_model/models/checkpoints/trainset_opensvdd'
            if not os.path.exists(openSVDD_save_path):
                os.makedirs(openSVDD_save_path)
            options.update(
                {
                    'train_typelist': train_split,  # [0,1,2,3,4,5,6]
                    'val_typelist': train_split,
                    'test_typelist': test_split,
                    'seed': 0,
                    'batch_size': 64,
                    'weight_path': weight_path,
                    'loss_path': loss_path,
                    'openSVDD_save_path': openSVDD_save_path,
                    'data_root': 'D:/wu/00data/',

                }
            )
            main(options)

