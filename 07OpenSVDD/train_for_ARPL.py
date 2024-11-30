import os
import argparse
import datetime
import time
import csv

import numpy as np

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset.dataset2 import dataset_for_image, dataset_for_radar, dataset_for_ADSB
from models.resnet50_for_1d import ResNet50
from utils.utils import Logger, save_networks, load_networks
from utils.utils import AverageMeter

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--data_root', type=str, default=r'.\data')
parser.add_argument('--dataset', type=str, default='radar')

# optimization
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--maxepoch', type=int, default=100)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model_name', type=str, default='ResNet1D')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="true", default=False)


# index
parser.add_argument('--index', type=str, default='0')
# threhold
parser.add_argument('--threhold', type=float, default=8.0)


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

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d_%H%M')
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60, '*'))


def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()

    loss_all = 0
    for batch_idx, (data, target_name, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x = net(data)
            logits, loss = criterion(x=x, y=None, labels=labels)
            loss.backward()
            optimizer.step()
        losses.update(loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg))
        loss_all += losses.avg
    return loss_all


def val(net, criterion, val_loader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0
    label_real = torch.tensor([])
    label_predict = torch.tensor([])
    torch.cuda.empty_cache()
    with torch.no_grad():
        for data, target_name, labels in val_loader:
            label_real = torch.cat((label_real, labels), 0)
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                y = net(data)
                logits, _ = criterion(x=y, y=None)
                predictions = logits.data.max(1)[1]
                label_predict = torch.cat((label_predict, predictions.cpu()), 0)

                total += labels.size(0)
                correct += (predictions == labels.data).sum()
        accuracy = 100 * float(correct / total)
        print('Accuracy on val set: %.2f %% [%d/%d]' % (accuracy, correct, total))
        cm = confusion_matrix(np.int64(label_real.cpu().numpy()), np.int64(label_predict.cpu().numpy()))
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


def main_worker(options):
    log_path = './log'
    make_print_to_file(path=log_path)
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']:
        use_gpu = False
    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

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
    feat_dim = len(options['train_typelist'])
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )

    # Model
    print("Creating model: {}".format(options['model_name']))
    if options['dataset'] == 'radar':
        model = ResNet50(in_channels=2, num_classes=len(options['train_typelist']))
    elif options['dataset'] == 'ADSB':
        model = ResNet50(in_channels=3, num_classes=len(options['train_typelist']))
    # Loss
    Loss = importlib.import_module('loss.' + options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    model_path = os.path.join(options['result_model_path'], 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = '{}_{}'.format(options['model_name'], options['loss'])  # index修改
    params_list = [{'params': model.parameters()},
                   {'params': criterion.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr=options['lr'])

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])

    start_time = time.time()
    best_osracc = 0
    for epoch in range(options['maxepoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['maxepoch']))
        train(model, criterion, optimizer, trainloader, epoch=epoch, **options)
        print("==> valid", options['loss'])
        results = val(model, criterion, validloader, epoch=epoch, **options)
        print("Acc (%): {:.3f}\t".format(results))
        if results >= best_osracc:
            best_osracc = results
            save_networks(model, model_path, file_name, criterion=criterion)
            print("********Save Net Successfully*********")
        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    results = dict()
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
        # for i in range(1):
            known = splits[options['dataset']][i]
            current_time = datetime.datetime.now()
            current_time = current_time.strftime('%Y%m%d%H%M')
            result_model_path = './log/ARPL/' + options['model_name'] + '/' + options['dataset'] + '/' + str(current_time) + '/result_model'
            confusion_matrix_path = './log/ARPL/' + options['model_name'] + '/' + options['dataset'] + '/' + str(current_time) + '/confusion_matrix'
            log_path = './log/ARPL/' + options['model_name'] + '/' + options['dataset'] + '/' + str(current_time)
            options.update(
                {
                    'train_typelist': known,  # [0,1,2,3,4,5,6]
                    'val_typelist': known,
                    'maxepoch': 100,
                    'seed': 0,
                    'lr': 0.0001,
                    'batch_size': 256,
                    'result_model_path': result_model_path,
                    'confusion_matrix_path': confusion_matrix_path,
                    'log_path': log_path,
                    'data_root': 'D:/wu/00data/',
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

            res = main_worker(options)

            end_time = time.time()
            execution_time = end_time - start_time
            print("Runtime：", execution_time, "s")

