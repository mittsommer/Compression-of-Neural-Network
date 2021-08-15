import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import *
import time
import pandas as pd
from pytorchtools import EarlyStopping

import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--l", default=0.999, help="lambda", type=float)
parser.add_argument("--gpu", default=False, type=bool)
parser.add_argument("--dataset", default='cifar10')
parser.add_argument("--model", default='Model_on_cifar10')
parser.add_argument("--topk", default=0.000001, type=float)
# parser.add_argument("--time", default=6, type=int)
CFG = parser.parse_args()

result_path = './result/{}/{}'.format(CFG.model, CFG.l)
if not os.path.exists(result_path):
    os.makedirs(result_path)


def train(net, epochs):
    train_losses = []
    valid_losses = []
    train_cross_entropies = []
    valid_cross_entropies = []
    avg_train_losses = []
    avg_valid_losses = []
    avg_train_cross_entropies = []
    avg_valid_cross_entropies = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True, path=result_path + '/checkpoint.pt')
    for epoch in range(1, epochs + 1):
        net.train()
        s = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net.forward(data).to(device)
            loss, cross_entropy, curvature = my_loss_function(output, target, batch_idx, epoch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_cross_entropies.append(cross_entropy)
            # print('batch_idx: {}/{} Curvature: {:.6f}'.format(batch_idx, len(train_loader), curvature))
        net.eval()  # prep model for evaluation
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            output = net(data).to(device)
            loss, cross_entropy, curvature = my_loss_function(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            valid_cross_entropies.append(cross_entropy)
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        train_cross_entropy = np.average(train_cross_entropies)
        valid_cross_entropy = np.average(valid_cross_entropies)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_train_cross_entropies.append(train_cross_entropy)
        avg_valid_cross_entropies.append(valid_cross_entropy)
        print(
            'Train Epoch: [{}/{}]  train_loss: {:.6f} valid_loss: {:.6f} train_CrossEntropy: {:.6f} '
            'valid_CrossEntropy: {:.6f} Curvature: {:.6f}'.format(
                epoch, epochs, train_loss, valid_loss, train_cross_entropy, valid_cross_entropy, curvature))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        train_cross_entropies = []
        valid_cross_entropies = []
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_cross_entropy, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        e = time.time()
        print("1 epoch time:{}".format(e - s))
    # load the last checkpoint with the best model
    if CFG.gpu:
        net.module.load_state_dict(torch.load(result_path + '/checkpoint.pt'))
    else:
        net.load_state_dict(torch.load(result_path + '/checkpoint.pt'))
    return avg_train_losses, avg_valid_losses, avg_train_cross_entropies, avg_valid_cross_entropies


def test(net):
    net.eval()
    test_loss = []
    curvature = 0
    crossentropy = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data).to(device)
        loss, cr, cu = my_loss_function(output, target)
        test_loss.append(loss.item())
        crossentropy += cr
        curvature += cu
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss = np.average(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    crossentropy = crossentropy / len(test_loader)
    curvature = curvature / len(test_loader)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset), accuracy))
    return test_loss, crossentropy, accuracy


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.index = torch.tensor([0])

    def forward(self, x, y, batch_idx=None, epoch=None):
        cross_entropy = CrossEntropy(x, y)
        # calculate first order derivative of all weights
        first_grad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
        if CFG.topk == 1.0:
            curvature = hessian(first_grad, net, device)
        else:
            curvature = hessian_topk(first_grad, net, CFG.topk)
        '''
        if batch_idx == len(train_loader):
            plot_hesse(hesse, epoch)'''
        return cross_entropy * CFG.l + curvature * (1 - CFG.l), cross_entropy.item(), curvature.item()


if __name__ == '__main__':
    print(CFG)
    device = setup(CFG.gpu)
    train_loader, test_loader, valid_loader = getData(CFG.dataset, CFG.batch_size)
    net = getModel(CFG.model, device)
    if CFG.gpu:
        net = nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(), CFG.learning_rate)
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    train_loss, valid_loss, train_cross_entropy, valid_cross_entropy = train(net, CFG.epoch)
    test(net)
    # plot_loss(train_loss, valid_loss, train_cross_entropy, valid_cross_entropy)
    if CFG.gpu:
        torch.save(net.module.state_dict(), result_path + '/{}_{}_hesse.pth'.format(CFG.dataset, CFG.model))
    else:
        torch.save(net.state_dict(), result_path + '/{}_{}_hesse.pth'.format(CFG.dataset, CFG.model))
