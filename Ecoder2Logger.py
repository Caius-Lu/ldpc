# ecoding:utf-8
from tensorboardX import SummaryWriter

class torch2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(torch2Logger, self).__init__(logdir)

    def log_training(self, split, train_loss, train_acc, learning_rate,
                    iteration):
        self.add_scalar("split {}: training loss".format(split), train_loss, iteration)
        self.add_scalar("split {}: train acc ".format(split), train_acc, iteration)
        self.add_scalar("split {}: slearning rate".format(split), learning_rate, iteration)

    def log_validation(self, split, val_acc, val_loss, iteration):
        self.add_scalar("split {}: validation  acc".format(split), val_acc, iteration)
        self.add_scalar("split {}: validation  loss".format(split), val_loss, iteration)
        