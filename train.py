'''*
 * @Author: caius.lu 
 * @Date: 2019-11-02 15:12:07 
 * @Last Modified by: caius.lu
 * @Last Modified time: 2019-11-02 15:31:57
 *'''

# ecoding:utf-8
import sys
import numpy as np
import time
import utils
import torch
from torch.optim import lr_scheduler

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
device = torch.device('cuda:0')
class Trainer:
    def __init__(self, model, optimizer, train_iter, val_iter, criteon,opt):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.criteon = criteon
        self.opt = opt### 
        self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
        self.start_time = time.time()


    def train(self, epoch):
        train_loss = 0
        train_acc = 0
        for i, (x,y) in enumerate(self.train_iter):
            x,y = x.to(device).float(),y.to(device).long()
            logits = self.model(x)
            loss = self.criteon(logits,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  
            # self.optimizer.use_cleargrads(use=False) 
            pred = logits.argmax(dim=1) ##??????
            train_acc += torch.eq(pred,y).sum().float().item()
            loss = float(loss.item()) 
            train_loss += loss
            elapsed_time = time.time() - self.start_time
            progress = (self.n_batches * (epoch - 1) + i + 1) * 1.0 / (self.n_batches * self.opt.nEpochs)
            eta = elapsed_time / progress - elapsed_time

            line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | loss: {} | Time: {} (ETA: {})'.format(
                epoch, self.opt.nEpochs, i + 1, self.n_batches,
                self.optimizer.param_groups[0]['lr'], loss/len(x), utils.to_hms(elapsed_time), utils.to_hms(eta))
            # logging.info(line)
            print(line)

        train_loss /= len(self.train_iter.dataset)
        train_acc = 100 * (train_acc / len(self.train_iter.dataset))

        return train_loss, train_acc ,self.optimizer.param_groups[0]['lr']

    def val(self):
        self.model.eval()
        val_acc = 0
        val_loss = 0
        # wrong = []
        # print(len(self.val_iter))
        for x, y in self.val_iter:
            x, y = x.to(device).float(), y.to(device).long()
            with torch.no_grad():
                logits = self.model(x)
                pred = logits.argmax(dim=1) 
            val_loss += self.criteon(logits,y)
            val_acc += torch.eq(pred,y).sum().float().item()
        print('================eval done========================')
        val_loss /= len(self.val_iter.dataset)
        val_acc = 100 * (val_acc / len(self.val_iter.dataset))
        return val_acc, val_loss
        # return val_top1, neg_acc, adult_cry_acc, child_cry_acc, angry_acc, neg_recall, adult_cry_recall, child_cry_recall, angry_recall


    # def lr_schedule(self):
    #     # divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
    #     # decay = sum(epoch > divide_epoch)
    #     # if epoch <= self.opt.warmup:
    #     #     decay = 1
    #     return lr_scheduler.StepLR(self.optimizer, 100, 0.1)  # # 每过10个epoch，学习率乘以0.1