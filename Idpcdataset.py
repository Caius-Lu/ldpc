
'''*
 * @Author: caius.lu 
 * @Date: 2019-11-02 15:12:15 
 * @Last Modified by: caius.lu
 * @Last Modified time: 2019-11-02 15:30:39
 *'''

import torch
import os
import glob
import random
import csv
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
root = '1023_all6/DVBS2ldpc_snr20'

class IDPC(Dataset):
    def __init__(self, signals, labels):
        super(IDPC, self).__init__()
        self.signals, self.labels = signals,labels
    def __len__(self):
        return len(self.signals)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x-mean)/std
        # x = x_hot*std+mean
        # x: [c,h,w]
        # mean:[3] ==> [3,1,1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std+mean
        return x

    def __getitem__(self, idx):
        # idx~[0-len(signals)]
        # self.iamges, self.labels
        sig, label = self.signals[idx], self.labels[idx]

        # tf = transforms.Compose([
        #     lambda x: Image.open(x).convert('RGB'),  # string path=> image data
        #     transforms.Resize(
        #         (int(self.resize*1.25), int(self.resize*1.25))),  # resize
        #     transforms.RandomRotation(15),  # 会出现黑色的
        #     transforms.CenterCrop(self.resize),  # 中心裁剪
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])  # imagenet 统计的均值和方差，具有代表性【-1,1】
        # ])
        # img = tf(img)
        sig = self.read_sig(sig)
        sig = torch.tensor(sig)
        label = torch.tensor(label)
        sig = torch.unsqueeze(sig,0)
        sig = torch.unsqueeze(sig,2)
        

        return sig, label
    
    def read_sig(self,sig_name): # 将原始信号转化为二进制数组形式
        si = open(sig_name,'r')
        si =si.read()
        si = si.split(',')
        si = np.array(si, np.float)
        si = si[:64800]
        return si

def load_csv(filename):
    if not os.path.exists(os.path.join(root, filename)):  # 当csv这个文件不存在，我们才创建它
        # signals = []
        signals = []
        labels = []
        for fd in os.listdir(root):
            if fd.endswith('.dat'):
                signals.append(os.path.join(root, fd))
        # C:\Users\wwy52\Desktop\文件\1023_all6\DVBS2ldpc_snr20
        # print(len(signals), signals)
        random.shuffle(signals)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for sig in signals:
                basename = os.path.basename(sig)
                label = basename.split('_')[0]
                # 'p\1023_all6\DVBS2ldpc_snr20\0_ldpc3_5_6.dat', 0
                writer.writerow([sig, label])
                print('ok')
            print('write into csv file:', filename)
    # read from csv file
    signals, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        # p\1023_all6\DVBS2ldpc_snr20\0_ldpc3_5_6.dat',0
        for row in reader:
            sig, label = row
            label = int(label)
            signals.append(sig)
            labels.append(label)
    assert len(signals) == len(labels)

    return signals, labels

def setup(opts, split):
    # 交叉验证
    signals, labels = load_csv('idpc.csv')
    if opts.mode == 'train':  
        signals = np.concatenate((
            signals[:int((split % opts.nFolds) * len(signals) / opts.nFolds)], 
                signals[int((split % opts.nFolds + 1) * len(signals) / opts.nFolds):]),axis = 0)
        labels =  np.concatenate((
            labels[:int((split % opts.nFolds) * len(labels) / opts.nFolds)], 
                labels[int((split % opts.nFolds + 1) * len(labels) / opts.nFolds):]),axis = 0)
        val_signals = signals[int((split % opts.nFolds) * len(signals) /opts.nFolds):
                                        int((split % opts.nFolds + 1) * len(signals) /opts.nFolds)]
        val_labels = labels[int((split % opts.nFolds) * len(labels) /opts.nFolds):
                                        int((split % opts.nFolds + 1) * len(labels) /opts.nFolds)]

        # Iterator setup
        train_data = IDPC(signals, labels)
        val_data = IDPC(val_signals, val_labels)
        return train_data, val_data
    elif opts.mode == 'test':  
        signals, labels = load_csv('idpc_test.csv')
        test_data = IDPC(signals, labels)
        return test_data



def main():
    import visdom
    import time
    import torchvision
    import os
    import argparse
    parser = argparse.ArgumentParser(description=' learning for ecoding')

    # General settings
    parser.add_argument('--split', type=int, default=1, help='esc: 1-5, urbansound: 1-10 (-1: run on all splits)')
    parser.add_argument('--nFolds', type=int, default=4)

    parser.add_argument('--mode', required=True, choices=['train', 'test'])

    # Testing settings

    opts = parser.parse_args()

    #多块使用逗号隔开
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    viz = visdom.Visdom()
    # db = torchvision.datasets.ImageFolder(root='pokeman',transform=tf)
    train, val = setup(opts, 1)
    x, y = next(iter(train))


    print('sample:', x.shape, y.shape, y)
    # viz.signals(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    loader = DataLoader(train, batch_size=32, shuffle=True,
                        num_workers=8)  # number_worker 多线程
    for x, y in loader:
        # viz.signals(db.denormalize(x), nrow=8,
        #             win='batch', opts=dict(title='batch'))
        # viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        # time.sleep(10)
        print(x.shape)
        print(y.shape)


if __name__ == '__main__':
    main()
