import torch.optim as optim
import argparse
from torch.nn.utils import prune
from utils import *
import time
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--fine_tune_epoch", default=10, type=int)
parser.add_argument("--learning_rate", default=0.003, type=float)
parser.add_argument("--l", default=1.0, help="lambda", type=float)
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--dataset", default='FashionMNIST')
parser.add_argument("--model", default='LeNet3_3')
parser.add_argument("--pruning_rate", default=0.6, type=float)
parser.add_argument("--time", default='1', type=str)
CFG = parser.parse_args()


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


def fine_tune(net, epochs):
    train_losses, valid_losses, train_cross_entropies, valid_cross_entropies = [], [], [], []
    avg_train_losses, avg_valid_losses, avg_train_cross_entropies, avg_valid_cross_entropies = [], [], [], []
    # initialize the early_stopping object
    for epoch in range(1, epochs + 1):
        net.train()
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
        net.eval()  # prep model for evaluation
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            output = net(data).to(device)
            loss, cross_entropy, curvature = my_loss_function(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            valid_cross_entropies.append(cross_entropy)
        train_loss, valid_loss = np.average(train_losses), np.average(valid_losses)
        train_cross_entropy, valid_cross_entropy = np.average(train_cross_entropies), np.average(valid_cross_entropies)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_train_cross_entropies.append(train_cross_entropy)
        avg_valid_cross_entropies.append(valid_cross_entropy)
        print('Train Epoch: [{}/{}]  train_loss: {:.6f} valid_loss: {:.6f} train_CrossEntropy: {:.6f} '
              'valid_CrossEntropy: {:.6f} Curvature: {:.6f}'.format(epoch, epochs, train_loss, valid_loss,
                                                                    train_cross_entropy, valid_cross_entropy,
                                                                    curvature))
        # clear lists to track next epoch
        train_losses, valid_losses, train_cross_entropies, valid_cross_entropies = [], [], [], []
        test(net)
    return avg_train_losses, avg_valid_losses, avg_train_cross_entropies, avg_valid_cross_entropies


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


def pruning(net, pruning_rate):
    # --------------Before pruning--------------
    print('Before pruning:')
    test(net)
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
    test(net)
    plot_all_weights(net, result_path + '/after.png')
    # ------------------Fine tune---------------------

    print('Fine tune:')
    fine_tune(net, CFG.fine_tune_epoch)
    print('After tune:')
    test_loss, crossentropy, accuracy = test(net)
    plot_all_weights(net, result_path + '/fine_tune.png')
    return accuracy


if __name__ == '__main__':
    sys.stdout = Logger(stream=sys.stdout)
    print(CFG)
    device = torch.device('cuda:0')
    train_loader, test_loader, valid_loader = getData(CFG.dataset, CFG.batch_size)
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    accs = []
    for p in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
        net = getModel(CFG.model, device)
        net.load_state_dict(torch.load(
            './result/{}_{}/time{}/lambda_{}/{}_{}_{}.pth'.format(CFG.model, CFG.dataset, CFG.time, CFG.l, CFG.dataset,
                                                                  CFG.model, CFG.l)))
        optimizer = optim.Adam(net.parameters(), CFG.learning_rate, weight_decay=0.0001)
        result_path = './result/{}_{}/time{}/lambda_{}/pruning_{}'.format(CFG.model, CFG.dataset, CFG.time, CFG.l, p)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        acc = pruning(net, p)
        accs.append(acc)
    print(accs)
