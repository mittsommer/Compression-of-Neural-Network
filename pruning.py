import torch
import torch.nn.utils.prune as prune
from models import LeNet5, LeNet3
import argparse
from utils import setup, getData, getModel
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--level", default=4)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--l", default=1.0, help="lambda", type=float)
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--dataset", default='cifar10')
parser.add_argument("--model", default='LeNet5')
parser.add_argument("--bits", default=8, type=int)

parser.add_argument("--pruning_rate",default=0.4,type=float)
CFG = parser.parse_args()


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
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset), accuracy))
    return test_loss, accuracy, crossentropy, curvature




if __name__ == '__main__':
    device = setup(CFG.device)
    train_loader, test_loader = getData(CFG.dataset, CFG.batch_size)
    net = getModel(CFG.model, device)
    net.load_state_dict(torch.load('./model/{}_{}_l={}_{}.pth'.format(CFG.dataset, CFG.model, CFG.l, CFG.epoch)))
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    pruning_rate = 0.6
    log_file = open("./pruning_log/pruning_log_MNIST_{}.log".format(pruning_rate), "a")
    # --------------Before pruning--------------
    print('Before pruning:', file=log_file)
    print('Before pruning:')
    test(net)
    # ---------------pruning----------------------
    parameters_to_prune = (
        (net.conv1, 'weight'),
        (net.conv2, 'weight'),
        (net.fc1, 'weight'),
        (net.fc2, 'weight'),
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )
    print('Pruning rate: {}'.format(pruning_rate), file=log_file)
    print('Pruning rate: {}'.format(pruning_rate))
    # ------------------After pruning---------------------
    print('After pruning:', file=log_file)
    print('After pruning:')
    test(net)
    log_file.close()
