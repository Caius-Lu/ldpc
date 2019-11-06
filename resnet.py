import  torch
from    torch import  nn
from    torch.nn import functional as F
import torchsummary
# from utils import Flatten



class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)

        return out




class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm2d(20)
        )
        # followed 4 blocks
        # [b, 16, h, w] => [b, 32, h ,w]
        self.blk1 = ResBlk(20, 40, stride=1)
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBlk(40, 80, stride=3)
        # # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(80, 160, stride=2)
        # # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(160, 320, stride=2)
        #self.blk5 = ResBlk(320, 640, stride=3)
        #self.blk6 = ResBlk(640, 1280, stride=2)
        self.avg = nn.AvgPool2d(5,5,2)
        self.drop = nn.Dropout(0.6)
        # [b, 256, 7, 7]
        self.outlayer = nn.Linear(320*360*1, num_class)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # print(x.shape)
        x = self.drop(self.avg(x))

        # print(x.shape)
        x = x.view(x.size(0), -1)

        # x = Flatten(x)
        x = self.outlayer(x)


        return x



def main():
    blk = ResBlk(64, 128)
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    print('block:', out.shape)


    model = ResNet18(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    torchsummary.summary(model.cuda(), (1, 64800, 1))
    tmp = torch.randn(2, 1, 64800, 1)
    out = model(tmp.to(device))
    print('resnet:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters())) # 总的参数量
    print('parameters size:', p)


if __name__ == '__main__':
    main()