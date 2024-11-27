import argparse
import datetime
import torch.nn.functional as F
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

from loss.DECLoss import regularization
from loss.centerloss import CenterLoss
from loss.SCLoss import SmoothL1CenterLoss
from loss.CCLoss import CCLoss
from dataset.dataset2 import dataset_for_image, dataset_for_radar, dataset_for_ADSB
from models.Rsenet18_1d import ResNet18
from models.SERsenet18_1d import SEResNet18
from models.SKResNet import SKResNet

from sklearn.preprocessing import LabelEncoder

from models.resnet50_for_1d import ResNet50

parser = argparse.ArgumentParser("Training")

# optimization
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--maxepoch', type=int, default=5)
parser.add_argument('--best_acc_val', type=float, default=0)   #记录在验证集上表现最佳的AUROC

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


def train(model, device, optimizer, train_loader, epoch,  **options):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    loss_list = []
    for batch_idx, (inputs, target_name, target) in enumerate(train_loader, 0):
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        features, centers, distance = model(inputs)
        loss1 = criterion(distance, target)
        loss2 = regularization(features, centers, target)
        loss = loss1 + options['reg'] * loss2
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print('[%d, %5d] Average loss over nearly 100 iterations: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0
    return loss_list


def val(model, device, val_loader, epoch, **options):
    model.eval()
    correct = 0
    total = 0
    label_real = torch.tensor([])
    label_predict = torch.tensor([])
    with torch.no_grad():
        for data in val_loader:
            inputs, target_name, target = data
            label_real = torch.cat((label_real, target), 0)
            inputs, target = inputs.to(device), target.to(device)
            features, centers, distance = model(inputs)
            _, predicted = torch.max(distance.data, dim=1)
            label_predict = torch.cat((label_predict, predicted.cpu()), 0)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * float(correct / total)
    print('Accuracy on val set: %d %% [%d/%d]' % (accuracy, correct, total))
    cm = confusion_matrix(np.int64(label_real.cpu().numpy()),np.int64(label_predict.cpu().numpy()))
    cm_name = 'Confusion_Matrix_' + 'epoch' + str(epoch) + '.png'
    cm_path = os.path.join(options['confusion_matrix_path'], cm_name)
    signal_list = options['train_typelist']
    plot_confusion_matrix(cm, cm_path, signal_list, title='Confusion Matrix')
    return accuracy


def plot_confusion_matrix(cm, savename, signal_list, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=300)
    np.set_printoptions(precision=2)
    num_classes = len(signal_list)
    ind_array = np.arange(num_classes)
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val] / np.sum(cm[y_val][:])
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(num_classes))

    plt.xticks(xlocations, signal_list, rotation=90)
    plt.yticks(xlocations, signal_list)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(num_classes)) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.close('all')
    # plt.show()


def save_networks(networks, resultpath, netname, finalloss):
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    weights = networks.state_dict()
    filename = '{}/{}.pth'.format(resultpath, netname)
    torch.save(weights, filename)
    return filename


def delete_networks(networks,resultpath,netname,finalloss):
    filename = '{}/{}.pth'.format(resultpath, netname)
    if os.path.isfile(filename):
        os.remove(filename)
    filename = '{}/{}_fullnet.pth'.format(resultpath, netname)
    if os.path.isfile(filename):
        os.remove(filename)


def main(options):
    seed_everything(options['seed'])
    log_path = options['log_path']
    make_print_to_file(path=log_path)
    data_root_path = options['data_root'] + options['dataset']
    # Dataset
    if options['dataset'] == 'radar':
        trainset = dataset_for_radar(data_root_path, options['train_typelist'], mode_type='train')
        validset = dataset_for_radar(data_root_path, options['train_typelist'], mode_type='val')
    else:
        trainset = dataset_for_ADSB(data_root_path,  options['train_typelist'], mode_type='train')
        validset = dataset_for_ADSB(data_root_path, options['train_typelist'], mode_type='val')

    #  dataloader
    trainloader = DataLoader(trainset, batch_size=options['batch_size'], shuffle=True, pin_memory=True)
    print("All training data:", len(trainset))
    validloader = DataLoader(validset, batch_size=options['batch_size'], shuffle=True, pin_memory=True)
    print("All validing data:", len(validset))

    options['num_classes'] = len(options['train_typelist'])
    # Model
    print("Creating model: {}".format(options['model_name']))
    if options['dataset'] == 'radar':
        model = ResNet50(in_channels=2, num_classes=len(options['train_typelist']))
    elif options['dataset'] == 'ADSB':
        model = ResNet50(in_channels=3, num_classes=len(options['train_typelist']))
    print(model)

    device = torch.device('cuda:0')
    model = model.to(device)

    lrate=options['lr']
    optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-4)

    params = list(model.parameters())
    total = sum([param.nelement() for param in params])
    print(options['model_name'])
    print("Number of parameter: %.2fM" % (total / 1e6))
    print(model)

    loss_train = []
    accuracy_val = []
    maxepoch = options['maxepoch']
    best_prec_val = torch.zeros(1)
    finalloss_before = []
    for epoch in range(maxepoch):
        print("=> training... epoch: ", epoch + 1)
        print('lr_this_epoch: ', optimizer.param_groups[0]['lr'])
        loss_train.extend(train(model, device, optimizer, trainloader, epoch, **options))
        prec_val = val(model, device, validloader, epoch, **options)
        accuracy_val.append(prec_val)
        is_best = bool(prec_val >= best_prec_val)
        if is_best:
            best_prec_val = prec_val
            finalloss = str("%.5f" % loss_train[-1])
            delete_networks(model, options['result_model_path'], netname=options['model_name'], finalloss=finalloss_before)
            pthFileFullName = save_networks(model, options['result_model_path'], netname=options['model_name'], finalloss=finalloss)
            finalloss_before = finalloss
            last_train_time = time.strftime("%m-%d_%H-%M", time.localtime())
    make_print_to_file(path='./')


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options.update({
        'model_name': 'ResNet50'
    })
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
            result_model_path = './log/PC/' + options['model_name'] + '/' + options['dataset'] + '/' + str(current_time) + '/result_model'
            confusion_matrix_path = './log/PC/' + options['model_name'] + '/' + options['dataset'] + '/' + str(current_time) + '/confusion_matrix'
            log_path = './log/PC/' + options['model_name'] + '/' + options['dataset'] + '/' + str(current_time)
            options.update(
                {
                    'train_typelist': known,
                    'val_typelist': known,
                    'maxepoch': 100,
                    'seed': 0,
                    'lr': 0.0001,
                    'batch_size': 256,
                    'reg': 0.001,
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
            print("Runtime：", execution_time, "s")
