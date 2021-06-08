import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from MyModel import MyModel
import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans, kmeans_predict

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
CFG = {
    'batch_size': 64,
    'lambda': 1,
    'iteration': 1
}

transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_data = CIFAR10('./data', train=False, download=True, transform=transformer)
test_loader = DataLoader(test_data, batch_size=CFG['batch_size'], shuffle=True)


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        cross_entropy = CrossEntropy(x, y)
        firstgrad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
        secondgrad = []
        for i, parm in enumerate(net.parameters()):
            secondgrad.append(torch.autograd.grad(firstgrad[i], parm, retain_graph=True, create_graph=True,
                                                  grad_outputs=torch.ones_like(firstgrad[i]))[0])
        curvature = torch.tensor(0)
        for i in secondgrad:
            curvature = curvature + torch.sum(torch.pow(i, 2))
        return cross_entropy * CFG['lambda'] + curvature * (1 - CFG['lambda']), cross_entropy, curvature

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


def kmeans_quantize(net):
    parm = list(net.parameters())
    weights = parm[0].view(-1)
    for i, p in enumerate(parm):
        if i == 0:
            continue
        if i % 2 == 0:
            weights = torch.cat([weights, p.view(-1)])
    weights = weights.reshape(-1, 1)
    print('weights number: ',weights.size()[0])
    # x = weights.to('cpu').detach().numpy().reshape(-1, 1)
    cluster_ids_x, cluster_centers = kmeans(
        X=weights, num_clusters=256, distance='euclidean', device=device, tol=0.0005
    )
    # kmeans_cluster = KMeans(n_clusters=256, max_iter=CFG['iteration'])
    # kmeans_cluster.fit(x)
    conv1_weights = net.conv1.weight
    conv1_labels = kmeans_predict(conv1_weights.view(-1,1), cluster_centers)
    conv1_quantized = cluster_centers[conv1_labels].reshape(conv1_weights.shape)
    net.conv1.weight = nn.parameter.Parameter(conv1_quantized.to(device))

    conv2_weights = net.conv2.weight
    conv2_labels = kmeans_predict(conv2_weights.view(-1,1), cluster_centers)
    conv2_quantized = cluster_centers[conv2_labels].reshape(conv2_weights.shape)
    net.conv2.weight = nn.parameter.Parameter(conv2_quantized.to(device))

    conv3_weights = net.conv3.weight
    conv3_labels = kmeans_predict(conv3_weights.view(-1,1), cluster_centers)
    conv3_quantized = cluster_centers[conv3_labels].reshape(conv3_weights.shape)
    net.conv3.weight = nn.parameter.Parameter(conv3_quantized.to(device))

    fc1_weights = net.fc1.weight
    fc1_labels = kmeans_predict(fc1_weights.view(-1,1), cluster_centers)
    fc1_quantized = cluster_centers[fc1_labels].reshape(fc1_weights.shape)
    net.fc1.weight = nn.parameter.Parameter(fc1_quantized.to(device))

    fc2_weights = net.fc2.weight
    fc2_labels = kmeans_predict(fc2_weights.view(-1,1), cluster_centers)
    fc2_quantized = cluster_centers[fc2_labels].reshape(fc2_weights.shape)
    net.fc2.weight = nn.parameter.Parameter(fc2_quantized.to(device))

    fc3_weights = net.fc3.weight
    fc3_labels = kmeans_predict(fc3_weights.view(-1,1), cluster_centers)
    fc3_quantized = cluster_centers[fc3_labels].reshape(fc3_weights.shape)
    net.fc3.weight = nn.parameter.Parameter(fc3_quantized.to(device))


if __name__ == '__main__':
    net = torch.load('./model/lambda={}.pth'.format(CFG['lambda']))
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    test(net)
    kmeans_quantize(net)
    test(net)
    print(CFG)
    # torch.save(net, './lambda={}_q_{}bit_{}.pth'.format(CFG['lambda'], CFG['bit'], CFG['iteration']))

    '''writer = SummaryWriter('./Result')
    for i, (name, param) in enumerate(net.named_parameters()):
        writer.add_histogram(name, param, 0)'''
