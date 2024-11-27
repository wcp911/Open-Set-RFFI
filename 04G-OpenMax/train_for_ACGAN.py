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
from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser("Training")

# optimization
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate for model")

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


def make_print_to_file(path='./log'):

    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60, '*'))


def train(g_model,d_model, binary_loss, auxiliary_loss,mse_criterion,g_optimizer,d_optimizer, train_loader, device, **options):
    batch_d_loss, batch_g_loss,batch_g_class_loss = 0,0,0
    for batch_idx, (inputs, target_name, target) in enumerate(train_loader, 0):
        inputs, target = inputs.to(device), target.to(device)
        real_label = torch.ones(len(inputs)).cuda()
        fake_label = torch.zeros(len(inputs)).cuda()

        binary_real_out,class_real_out = d_model(inputs)
        real_label = real_label.reshape([-1, 1])
        dloss_binary_real = binary_loss(binary_real_out, real_label)
        dloss_class_real = auxiliary_loss(class_real_out,target)
        noise = torch.randn(len(inputs), 100).cuda()
        fake_img = g_model(noise, target).detach()
        binary_fake_out, class_fake_out = d_model(fake_img)
        fake_label = fake_label.reshape([-1, 1])
        dloss_binary_fake = binary_loss(binary_fake_out, fake_label)
        dloss_class_fake = auxiliary_loss(class_fake_out, target)

        d_loss = (dloss_binary_real+dloss_binary_fake) + (dloss_class_real+dloss_class_fake)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        noise = torch.randn(len(inputs), 100).cuda()

        fake_img = g_model(noise, target)
        output1, output2 = d_model(fake_img)
        gloss_binary = binary_loss(output1, real_label)
        gloss_class = auxiliary_loss(output2, target)
        g_loss = gloss_binary+gloss_class

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        batch_d_loss += d_loss
        batch_g_loss += g_loss
        batch_g_class_loss += gloss_class

    # if d_loss < 0.5:
    #     for i in d_optimizer.param_groups:
    #         i['lr']=lr_d/10

    for g_or_d, g_d_name in zip([g_model, d_model], ['_g_', '_d_']):
        torch.save(g_or_d, os.path.join(options['result_model_path'], g_d_name + 'last.pth'))
    return batch_g_loss,batch_d_loss,batch_g_class_loss


def main(options):
    seed_everything(options['seed'])
    log_path = options['log_path']
    make_print_to_file(path=log_path)
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
    # Model
    print("Creating model: {}".format(options['model_name']))
    if options['dataset'] == 'radar':
        netD = Discriminator(in_channels=2, num_class=len(options['train_typelist']))
        netG = Generator(in_channels=2, num_class=len(options['train_typelist']))
    else:
        netD = Discriminator(in_channels=3, num_class=len(options['train_typelist']))
        netG = Generator(in_channels=3, num_class=len(options['train_typelist']))

    device = torch.device('cuda:0')
    netD = netD.to(device)
    netG = netG.to(device)

    # loss functions
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    # setup optimizer
    lr_d = 0.0001
    lr_g = 0.0001
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))

    loss_list_g, loss_list_d, loss_list_g_class = [], [], []
    maxepoch = options['maxepoch']

    for epoch in range(maxepoch):
        print("=> training... epoch: ", epoch + 1)
        result = train(netG, netD, dis_criterion, aux_criterion, mse_criterion, optimizerG, optimizerD, trainloader, device, **options)
        batch_g_loss, batch_d_loss, batch_g_class_loss = result
        loss_list_g.append(batch_g_loss.item() / len(trainloader))
        loss_list_d.append(batch_d_loss.item() / len(trainloader))
        loss_list_g_class.append(batch_g_class_loss.item() / len(trainloader))
        print('\nEpoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} ,g_loss_class:{:.6f}'
              .format(epoch, maxepoch,
                      batch_d_loss.item() / len(trainloader),
                      batch_g_loss.item() / len(trainloader),
                      batch_g_class_loss.item() / len(
                          trainloader)))

        plt.plot(range(len(loss_list_g)), loss_list_g, label="g_loss")
        plt.plot(range(len(loss_list_d)), loss_list_d, label="d_loss")
        plt.plot(range(len(loss_list_g_class)), loss_list_g_class, label="g_class_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(options['log_path'], '/loss.jpg'))
        plt.clf()
        torch.cuda.empty_cache()
    make_print_to_file(path='./')


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)

    from split import train_splits as splits
    train_list = ['radar', 'ADSB']
    for j in range(len(train_list)):
        torch.cuda.empty_cache()
        options.update(
            {
                'dataset': train_list[j]
            }
        )
        for i in range(len(splits[options['dataset']])):
            known = splits[options['dataset']][i]
            current_time = datetime.datetime.now()
            current_time = current_time.strftime('%Y%m%d%H%M')
            result_model_path = './log/ACGAN/' + options['dataset'] + '/' + str(current_time) + '/result_model'
            confusion_matrix_path = './log/ACGAN/'+ options['dataset'] + '/' + str(current_time) + '/confusion_matrix'
            log_path = './log/ACGAN/'+ options['dataset'] + '/' + str(current_time)
            options.update(
                {
                    'train_typelist': known,  # [0,1,2,3,4,5,6]
                    'val_typelist': known,
                    'maxepoch': 300,
                    'seed': 0,
                    'lr': 0.0001,
                    'batch_size': 64,
                    'result_model_path': result_model_path,
                    'confusion_matrix_path': confusion_matrix_path,
                    'log_path': log_path,
                    'data_root': 'D:/wu/00data/'
                }
            )
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            if not os.path.exists(result_model_path):
                os.makedirs(result_model_path)
            if not os.path.exists(confusion_matrix_path):
                os.makedirs(confusion_matrix_path)

            make_print_to_file(path=log_path)

            start_time = time.time()

            res = main(options)

            end_time = time.time()
            execution_time = end_time - start_time
            print("Runtime：", execution_time, "S")
