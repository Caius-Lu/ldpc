import torch
from torch import optim,nn
import visdom
import torch
from torch.utils.data import DataLoader
import os
from Idpcdataset_1 import LDPC
import torchsummary
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
import os

from torch.utils.data import DataLoader
from resnet import  ResNet18
import torchsummary
import Idpcdataset_1
from train import Trainer
import logging
from torch import optim
import utils
import torch
import _init_paths
from torch import nn
from torch.optim import lr_scheduler
from Ecoder2Logger import torch2Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

logging.basicConfig(level=logging.DEBUG,
                         filename='output.log',
                         datefmt='%Y/%m/%d %H:%M:%S',
                         format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

def evalute(model,loader):
        model.eval()
        print('eval : ======================')
        eval_step = 0

        correct = 0
        total = len(loader.dataset)
        # x;[b,3,224,224] y=[b]
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device)
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1) ##??????
            correct += torch.eq(pred,y).sum().float().item()
        print('================eval done========================')
        return correct/total




def main():
    test_db = LDPC('1023_all6/DVBS2ldpc_snr20_test1')
    print(len(test_db))
    test_loader = DataLoader(test_db, batch_size=64, num_workers=4)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = ResNet18(3).to(device)
    model.load_state_dict(torch.load('best1.mdl'))
    model.eval()
    logger.info('loaded from ckpt!')
    test_acc = evalute(model, test_loader)
    print()
    logger.info('test acc{}'.format(test_acc))
    print('test acc:',test_acc)

if __name__ == '__main__':
    main()