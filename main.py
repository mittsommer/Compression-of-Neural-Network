import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.functional import hessian
import argparse
from utils import setup, getModel, getData
import pyhessian
import seaborn as sns
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--l", default=0.9999, help="lambda", type=float)
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--dataset", default='cifar10')
parser.add_argument("--model", default='LeNet5')
CFG = parser.parse_args()


def train(net, epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = net.forward(data).to(device)
        loss, cross_entropy, curvature = my_loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f} CrossEntropy: {:.6f} Curvature: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
            loss.item(), cross_entropy, curvature))


def test(net):
    net.eval()
    test_loss = 0
    curvature = 0
    crossentropy = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data).to(device)
        t, cr, cu = my_loss_function(output, target)
        test_loss += t.item()
        crossentropy += cr.item()
        curvature += cu.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    crossentropy = crossentropy / len(test_loader)
    curvature = curvature / len(test_loader)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset), accuracy))
    return test_loss, accuracy, crossentropy, curvature


def eval_hessian(first_grad, model):
    cnt = 0
    for g in first_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    weights_number = g_vector.size(0)
    hessian_matrix = torch.zeros(weights_number, weights_number)
    for idx in range(weights_number):
        second_grad = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True, retain_graph=True)
        cnt = 0
        for g in second_grad:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian_matrix[idx] = g2
    return hessian_matrix


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        cross_entropy = CrossEntropy(x, y)
        # calculate first order grad derivative of all weights
        first_grad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
        # calculate second order grad derivative of every weight -- Hesse matrix diagonal
        second_grad = []
        for i, parm in enumerate(net.parameters()):
            second_grad.append(torch.autograd.grad(first_grad[i], parm, retain_graph=True, create_graph=True,
                                                   grad_outputs=torch.ones_like(first_grad[i]))[0])
        '''s = torch.tensor([]).to(device)
        for i in second_grad:
            s = torch.cat((s, i.view(-1)))
        s = s.cpu().detach().numpy()'''
        # calculate sum((d2L/dw2)^2)
        curvature = torch.tensor(0)
        for i in second_grad:
            curvature = curvature + torch.sum(torch.pow(i,2))
        # calculate whole Hesse matrix
        # hesse = eval_hessian(first_grad, net)
        '''curvature = torch.sum(torch.pow(hesse, 2))'''
        # (evals, evecs) = torch.eig(hesse,eigenvectors=True)
        # return cross_entropy * CFG.l + curvature * (1 - CFG.l), cross_entropy, curvature
        return cross_entropy * CFG.l + curvature * (1 - CFG.l), cross_entropy, curvature


if __name__ == '__main__':
    '''file = open('./log.log', 'a')
    print(CFG, file=file)'''
    device = setup(CFG.device)
    train_loader, test_loader = getData(CFG.dataset, CFG.batch_size)
    net = getModel(CFG.model, device)
    optimizer = optim.Adam(net.parameters(), CFG.learning_rate)
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    for epoch in range(CFG.epoch):
        train(net, epoch)
    test_loss, accuracy, crossentropy, curvature = test(net)
    # file.close()
    torch.save(net.state_dict(), './model/{}_{}_l={}_{}.pth'.format(CFG.dataset, CFG.model, CFG.l, CFG.epoch))
