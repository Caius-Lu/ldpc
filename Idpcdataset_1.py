import torch
import os
import glob
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class LDPC(Dataset):
    def __init__(self, root):
        super(LDPC, self).__init__()
        self.root = root
        self.signals, self.labels = self.load_csv('idpc.csv')

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):  # 当csv这个文件不存在，我们才创建它
            # signals = []
            signals = []
            labels = []
            for fd in os.listdir(self.root):
                if fd.endswith('.dat'):
                    signals.append(os.path.join(self.root, fd))
                    # C:\Users\wwy52\Desktop\文件\1023_all6\DVBS2ldpc_snr20
            # print(len(signals), signals)
            # random.shuffle(signals)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
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
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            # p\1023_all6\DVBS2ldpc_snr20\0_ldpc3_5_6.dat',0
            for row in reader:
                sig, label = row
                label = int(label)
                signals.append(sig)
                labels.append(label)
        assert len(signals) == len(labels)

        return signals, labels

    def __len__(self):
        return len(self.signals)

    # def denormalize(self, x_hat):
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    #     # x_hat = (x-mean)/std
    #     # x = x_hot*std+mean
    #     # x: [c,h,w]
    #     # mean:[3] ==> [3,1,1]
    #     mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    #     std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    #
    #     x = x_hat * std + mean
    #     return x

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
        sig = torch.unsqueeze(sig, 0)
        sig = torch.unsqueeze(sig, 2)

        return sig, label

    def read_sig(self, sig_name):  # 将原始信号转化为二进制数组形式
        si = open(sig_name, 'r')
        si = si.read()
        si = si.split(',')
        si = np.array(si, np.float)
        si = si[:64800]
        return si


def main():
    import visdom
    import time
    import torchvision
    import os

    # 多块使用逗号隔开
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    viz = visdom.Visdom()
    # db = torchvision.datasets.ImageFolder(root='pokeman',transform=tf)
    db = LDPC('/media/data1/ldpc/1023_all6/DVBS2ldpc_snr20', 'train')
    x, y = next(iter(db))

    print('sample:', x.shape, y.shape, y)
    # viz.signals(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    loader = DataLoader(db, batch_size=32, shuffle=True,
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
