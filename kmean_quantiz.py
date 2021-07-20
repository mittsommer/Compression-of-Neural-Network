import torch
import torch.nn as nn
from kmeans_pytorch import kmeans, kmeans_predict
import operator
import argparse
from utils import *
from plot_weights import plot_all_weights
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--level", default=2)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--l", default=1.0, help="lambda", type=float)
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--dataset", default='MNIST')
parser.add_argument("--model", default='LeNet3')
CFG = parser.parse_args()


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        cross_entropy = CrossEntropy(x, y)
        # calculate first order grad derivative of all weights
        first_grad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
        # calculate second order grad derivative of every weight -- Hesse matrix diagonal
        '''second_grad = []
        for i, parm in enumerate(net.parameters()):
            second_grad.append(torch.autograd.grad(first_grad[i], parm, retain_graph=True, create_graph=True,
                                                   grad_outputs=torch.ones_like(first_grad[i]))[0])
        curvature = torch.tensor(0)
        for i in second_grad:
            curvature = curvature + torch.sum(torch.pow(i,2))'''
        # calculate whole Hesse matrix
        hesse = eval_hessian(first_grad, net)
        curvature = torch.sum(torch.pow(hesse, 2))

        return cross_entropy * CFG.l + curvature * (1 - CFG.l), cross_entropy, curvature


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
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset), accuracy))
    return test_loss, accuracy, crossentropy, curvature


def kmeans_quantize(net, level):
    parm = net.parameters()
    for i, p in enumerate(parm):
        if i == 0:
            weights = p.data.view(-1, 1)
        else:
            weights = torch.cat((weights, p.data.view(-1, 1)))
    cluster_ids_x, cluster_centers = kmeans(X=weights, num_clusters=level, device=device, tol=0.0001)

    parm_ = net.named_parameters()
    for name, param in parm_:
        netweight = operator.attrgetter(name)(net)
        netweight_label = kmeans_predict(netweight.reshape(-1, 1), cluster_centers, device=device)
        netweight_quanti = cluster_centers[netweight_label].reshape(netweight.size())
        net_change = operator.attrgetter(name)(net)
        net_change.data.copy_(nn.parameter.Parameter(netweight_quanti.type(torch.cuda.FloatTensor)))


if __name__ == '__main__':
    acc = []
    for i in range(10):
        device = setup(CFG.device)
        train_loader, test_loader, valid_loader = getData(CFG.dataset, CFG.batch_size)
        net = getModel(CFG.model, device)
        net.load_state_dict(torch.load('./model/{}_{}_l={}_hesse.pth'.format(CFG.dataset, CFG.model, CFG.l, CFG.epoch)))
        # plot_all_weights(net)
        my_loss_function = MyLoss()
        CrossEntropy = nn.CrossEntropyLoss()
        # test(net)
        kmeans_quantize(net, CFG.level)
        print("K-means level:{}, After k-means quantize:".format(CFG.level))
        test_loss, accuracy, crossentropy, curvature = test(net)
        acc.append(accuracy)
        # plot_all_weights(net)
        # torch.save(net, './lambda={}_q_{}bit.pth'.format(CFG.l, CFG.bit))
    print(acc)
