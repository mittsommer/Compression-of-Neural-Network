import torch.optim as optim
import argparse
from torch.nn.utils import prune
from utils import *
import time
import os, sys
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--fine_tune_epoch", default=30, type=int)
parser.add_argument("--learning_rate", default=0.003, type=float)
parser.add_argument("--l", default=0.999, help="lambda", type=float)
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--dataset", default='FashionMNIST')
parser.add_argument("--model", default='LeNet3_3')
# parser.add_argument("--pruning_rate", default=0.475, type=float)
parser.add_argument("--time", default='1', type=str)
CFG = parser.parse_args()


# 0.475 0.595 0.685 0.715 0.735 0.755 0.775 0.815 0.855

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, batch_idx=None, epoch=None, is_test=False):
        cross_entropy = CrossEntropy(x, y)
        if CFG.l == 1.0:
            curvature = torch.tensor(0)
        else:
            # calculate first order derivative of all weights
            first_grad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
            hesse = hessian(first_grad, net, device)
            curvature = torch.sum(torch.pow(hesse, 2))
        return cross_entropy * CFG.l + curvature * (1 - CFG.l), cross_entropy.item(), curvature.item()


def test(net):
    net.eval()
    test_loss = []
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data).to(device)
        loss = nn.CrossEntropyLoss()(output, target)
        test_loss.append(loss.item())
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss = np.average(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset), accuracy))
    return test_loss, accuracy


def fine_tune(net, epochs, pruning_result):
    train_losses, valid_losses = [], []
    avg_train_losses, avg_valid_losses = [], []
    # initialize the early_stopping object
    for epoch in range(1, epochs + 1):
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net.forward(data).to(device)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        net.eval()  # prep model for evaluation
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            output = net(data).to(device)
            loss = nn.CrossEntropyLoss()(output, target)
            # record validation loss
            valid_losses.append(loss.item())
        train_loss, valid_loss = np.average(train_losses), np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print(
            'Train Epoch: [{}/{}]  train_loss: {:.6f} valid_loss: {:.6f}'.format(epoch, epochs, train_loss, valid_loss))
        # clear lists to track next epoch
        train_losses, valid_losses = [], []
        test_loss, accuracy = test(net)
        if epoch == 5 or epoch == 30:
            pruning_result.append(train_loss)
            pruning_result.append(accuracy)

    return pruning_result


class Logger(object):
    def __init__(self,
                 filename='./result/{}_{}/time{}/lambda_{}/pruning.log'.format(CFG.model, CFG.dataset, CFG.time, CFG.l),
                 stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def pruning(net, pruning_rate, pruning_result):
    # --------------Before pruning--------------
    print('Before pruning:')
    test_loss, accuracy = test(net)

    pruning_result.append(accuracy)

    plot_all_weights(net, result_path + '/before.png')
    # ---------------pruning----------------------
    parameters_to_prune = (
        (net.conv1, 'weight'),
        (net.conv2, 'weight'),
        (net.fc1, 'weight')
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )
    print('Pruning rate: {}'.format(pruning_rate))
    test_loss, accuracy = test(net)

    pruning_result.append(accuracy)

    plot_all_weights(net, result_path + '/after.png')
    # ------------------Fine tune---------------------

    print('Fine tune:')
    pruning_result = fine_tune(net, CFG.fine_tune_epoch, pruning_result)
    print('After tune:')
    test_loss, accuracy = test(net)
    # plot_all_weights(net, result_path + '/fine_tune.png')
    return pruning_result


if __name__ == '__main__':
    sys.stdout = Logger(stream=sys.stdout)
    print(CFG)
    device = torch.device('cuda:0')
    train_loader, test_loader, valid_loader = getData(CFG.dataset, CFG.batch_size)
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    accs5 = []
    accs = []
    if CFG.model == 'LeNet3_3':
        pruning_rate = [0.62, 0.72, 0.78, 0.83, 0.87, 0.88, 0.89, 0.90, 0.92, 0.94, 0.97]
        #
    elif CFG.model == 'LeNet3_2':
        pruning_rate = [0.55, 0.68, 0.76, 0.82, 0.86, 0.88, 0.89, 0.90, 0.91, 0.93, 0.97]
    elif CFG.model == 'LeNet3':
        pruning_rate = [0.23, 0.52, 0.68, 0.78, 0.84, 0.86, 0.88, 0.89, 0.91, 0.93, 0.96]
    pruning_results = []
    for p in pruning_rate:
        pruning_result = []

        net = getModel(CFG.model, device)
        net.load_state_dict(torch.load(
            './result/{}_{}/time{}/lambda_{}/{}_{}_{}.pth'.format(CFG.model, CFG.dataset, CFG.time, CFG.l, CFG.dataset,
                                                                  CFG.model, CFG.l)))
        optimizer = optim.Adam(net.parameters(), CFG.learning_rate, weight_decay=0.0001)
        result_path = './result/{}_{}/time{}/lambda_{}/pruning_{}'.format(CFG.model, CFG.dataset, CFG.time, CFG.l, p)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        pruning_result = pruning(net, p, pruning_result)
        pruning_results.append(pruning_result)

    result = pd.DataFrame(pruning_results,
                          columns=['acc_before', 'acc_after', 'finetune5_loss', 'finetune5_acc', 'finetune30_loss',
                                   'finetune30_acc'],
                          index=pruning_rate)
    result.to_csv('./result/{}_{}/time{}/lambda_{}/pruning.csv'.format(CFG.model, CFG.dataset, CFG.time, CFG.l))
