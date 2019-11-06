# encoding: utf-8
'''/*
 * @Author: caius.lu 
 * @Date: 2019-11-02 14:32:02 
 * @Last Modified by: caius.lu
 * @Last Modified time: 2019-11-02 15:30:14
 */'''


import sys
import os 
import argparse
import Idpcdataset 
from torch.utils.data import DataLoader
from resnet import  ResNet18
import torchsummary
import Idpcdataset
from train import Trainer
import logging
from torch import optim
import utils
import torch
import _init_paths
from torch import nn
from torch.optim import lr_scheduler
from Ecoder2Logger import torch2Logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
best_acc = 0

def main():

    parser = argparse.ArgumentParser(description='deep learning for sounds')

    # General settings
    # parser.add_argument('--dataset', required=True, choices=['audio', 'esc50', 'urbansound8k'])
    # parser.add_argument('--netType', required=True, choices=['envnet', 'envnetv2'])
    # parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--split', type=int, default=-1, help='esc: 1-5, urbansound: 1-10 (-1: run on all splits)')
    parser.add_argument('--save', default='None', help='Directory to save the results')
    parser.add_argument('--testOnly', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--nFolds', type=int, default=4)

    # # Learning settings (default settings are defined below)
    # parser.add_argument('--BC', action='store_true', help='BC learning')
    # parser.add_argument('--strongAugment', action='store_true', help='Add scale and gain augmentation')
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--LR', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--schedule', type=float, nargs='*', default=-1, help='When to divide the LR')
    parser.add_argument('--warmup', type=int, default=-1, help='Number of epochs to warm up')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--weightDecay', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--mode', required=True, choices=['train', 'test'],default='train')
    parser.add_argument('--num_class', type=int, default=3)

    # Testing settings
    parser.add_argument('--restore',  action='store_true', default=True)


    opt = parser.parse_args()
    utils.display_info(opt)
    model = ResNet18(opt.num_class).to(device)
    torchsummary.summary(model.cuda(), (1, 64800, 1))
    log_writer = torch2Logger('logs')
    if opt.restore:
        model.load_state_dict(torch.load('best.mdl'))
    splits = range(1, opt.nFolds + 1)
    for split  in splits:
        print('+-- Split {} --+'.format(split))
        train(opt, split, model, log_writer=log_writer)
    log_writer.close()

def train(opt, split, model, log_writer):

    optimizer = optim.Adam(model.parameters(),lr = 1e-6,weight_decay=0.001)
    criteon = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)  # # 每过100个epoch，学习率乘以0.1
    train_dataset, val_datset = Idpcdataset.setup(opt, split)
    train_iter = DataLoader(train_dataset,batch_size=opt.batchSize,shuffle=True,num_workers=4)
    val_iter = DataLoader(val_datset,batch_size=opt.batchSize,num_workers=4)
    trainer = Trainer(model, optimizer, train_iter, val_iter,criteon, opt)


    for epoch in range(1, opt.nEpochs + 1):
        scheduler.step()
        train_loss, train_acc ,learning_rate= trainer.train(epoch)
        log_writer.log_training(split, train_loss, train_acc, learning_rate, epoch)
        val_acc, val_loss = trainer.val()
        log_writer.log_validation(split, val_acc, val_loss, epoch)
        global best_acc
        if val_acc>best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(),'best.mdl')


        msg = '| Epoch: {}/{} | Train: LR {}  Loss {:.3f} train acc  {:.2f} | Val: acc: {:.2f}\n'.format(
                epoch, opt.nEpochs, trainer.optimizer.param_groups[0]['lr'], train_loss, train_acc, val_acc)
        # logging.info(msg)
        print(msg)
        if epoch % 100 == 0:
            torch.save(model.state_dict(),'model_train_split_{}_epoch_{}.npz'.format(split,epoch))


if __name__ == "__main__":
    main()