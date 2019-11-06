import torch
from torch import optim,nn
import visdom
import torch
from torch.utils.data import DataLoader
import os
from Idpcdataset import IDPC
from resnet import  ResNet18
import torchsummary
import torch.optim as optim
from torch.optim import lr_scheduler




os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batchsz =64
lr = 1e-6
epochs = 1000

device = torch.device('cuda')
torch.manual_seed(1234) # 设置随机种子 

train_db = IDPC('1023_all6/DVBS2ldpc_snr20', mode='train')
val_db = IDPC('1023_all6/DVBS2ldpc_snr20',mode='val')
test_db = IDPC('1023_all6/DVBS2ldpc_snr20',mode='test')
train_loader = DataLoader(train_db,batch_size=batchsz,shuffle=True,num_workers=4)
test_loader = DataLoader(test_db,batch_size=batchsz,num_workers=4)
val_loader = DataLoader(val_db,batch_size=batchsz,num_workers=4)

def evalute(model,loader,mode='test',epoch =0):
    if mode =='test':
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
    else:
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
                pred = logits.argmax(dim=1)  ##??????
            correct += torch.eq(pred, y).sum().float().item()
        print('epoch:     ',epoch,'      training  acc: ',correct / total)


'''

    #多块使用逗号隔开
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    viz = visdom.Visdom()
    # db = torchvision.datasets.ImageFolder(root='pokeman',transform=tf)
    db = IDPC('/media/data1/ldpc/1023_all6/DVBS2ldpc_snr20', 'training')
    x, y = next(iter(db))

    print('sample:', x.shape, y.shape, y)
    # viz.signals(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    loader = DataLoader(db, batch_size=32, shuffle=True,
                        num_workers=8)  # number_worker 多线程
'''

def main():
    viz = visdom.Visdom()
    model = ResNet18(3).to(device)
    model.load_state_dict(torch.load('best.mdl'))
    torchsummary.summary(model.cuda(), (1, 64800, 1))
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=0.01)
    criteon = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)  # # 每过10个epoch，学习率乘以0.1
    best_epoch,best_acc=0,0
    global_step = 0
    viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        print('epoch : ', epoch)
        scheduler.step()
        for step,(x,y) in enumerate(train_loader):
            # x;[b,3,224,224] y=[b]
            x,y = x.to(device).float(),y.to(device)
            logits = model(x)
            loss = criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
            print('global_step:',global_step, 'loss',loss.item())
        if epoch %1==0:
            evalute(model, train_loader,mode='train',epoch=epoch)
            val_acc = evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(),'best.mdl')
            viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc:',best_acc, 'best epoch',best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')
    test_acc = evalute(model,test_loader)
    print('test acc', test_acc)







if __name__ == '__main__':
    main()
