import torch
import torch.nn as nn
from kmeans_pytorch import kmeans, kmeans_predict
from kmeans_pytorch.
import operator
import argparse
from utils import *
from plot_weights import plot_all_weights
import seaborn as sns
import matlab
import matlab.engine

parser = argparse.ArgumentParser()
parser.add_argument("--level", default=256)
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

    def forward(self, x, y, batch_idx=None, epoch=None):
        cross_entropy = CrossEntropy(x, y)
        # calculate first order grad derivative of all weights
        first_grad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
        # calculate second order grad derivative of every weight -- Hesse matrix diagonal
        hesse = eval_hessian(first_grad, net, device)
        curvature = torch.sum(torch.pow(hesse, 2))
        return cross_entropy * CFG.l + curvature * (1 - CFG.l), cross_entropy.item(), curvature.item()


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


def max_lloyd_quantize(net, level):
    parm = net.parameters()
    for i, p in enumerate(parm):
        if i == 0:
            weights = p.data.view(-1, 1)
        else:
            weights = torch.cat((weights, p.data.view(-1, 1)))
    matlab_engine = matlab.engine.start_matlab()
    weights = matlab.double(weights.cpu().detach().numpy().reshape(-1).tolist())
    partition, codebook, distor, reldistor = matlab_engine.lloyds(weights, matlab.double([float(level)]), nargout=4)
    # cluster_ids_x, cluster_centers = kmeans(X=weights, num_clusters=level, device=device, tol=0.0001)

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
        net.load_state_dict(torch.load('./model/{}/{}_{}_hesse.pth'.format(CFG.l,CFG.dataset,CFG.model)))
        # plot_all_weights(net)
        my_loss_function = MyLoss()
        CrossEntropy = nn.CrossEntropyLoss()
        # test(net)
        max_lloyd_quantize(net, CFG.level)
        print("K-means level:{}, After k-means quantize:".format(CFG.level))
        test_loss, accuracy, crossentropy, curvature = test(net)
        acc.append(accuracy)
        # plot_all_weights(net)
        # torch.save(net, './lambda={}_q_{}bit.pth'.format(CFG.l, CFG.bit))
    print(acc)
