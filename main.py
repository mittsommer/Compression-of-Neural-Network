import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import *
import time
import pandas as pd
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--epoch", default=1, type=int)
parser.add_argument("--l", default=1.0, help="lambda", type=float)
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--dataset", default='MNIST')
parser.add_argument("--model", default='LeNet3')
parser.add_argument("--time", default=1, type=int)
CFG = parser.parse_args()

result_path = './result/{}/{}'.format(CFG.time, CFG.l)

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
    early_stopping = EarlyStopping(patience=10, verbose=True, path=result_path+'/checkpoint.pt')
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
    # load the last checkpoint with the best model
    net.load_state_dict(torch.load(result_path+'/checkpoint.pt'))
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

    def forward(self, x, y, batch_idx=None, epoch=None):
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
        # time_start = time.time()
        hesse = eval_hessian(first_grad, net, device)
        # time_end = time.time()
        # print("hesse time:{}".format(time_end-time_start))
        curvature = torch.sum(torch.pow(hesse, 2))
        # curvature = torch.tensor(0)
        '''curvature = torch.sum(torch.pow(
            torch.topk(torch.abs(hesse.view(-1)), k=int(hesse.shape[0]**2 * 0.005))[0], 2
        ))'''
        '''max = torch.max(torch.abs(hesse))*0.95
        for i in enumerate(hesse):
            for j in i:
                if torch.abs(j) >= max:
                    curvature = curvature + torch.pow(j, 2)'''
        if batch_idx == 187:
            plot_hesse(hesse, epoch)
        return cross_entropy * CFG.l + curvature * (1 - CFG.l), cross_entropy.item(), curvature.item()


def plot_hesse(hesse, epoch):
    ax = sns.distplot((hesse.flatten()).cpu().detach().numpy(), color='b', kde=False, bins=20, hist_kws={'log': True})
    ax.set_xlabel('Values of second derivatives')
    ax.set_ylabel('frequences')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(10, 400000)
    dist_fig = ax.get_figure()
    dist_fig.savefig(result_path+'/pics/hesse_distribution_{}.png'.format(epoch))
    plt.close()
    bx = sns.heatmap(hesse.cpu().detach().numpy(), vmin=-0.5, vmax=0.5)
    bx.set_xlabel('row')
    bx.set_ylabel('column')
    plt.title('Hesse Matrix')
    ax.xaxis.tick_top()
    heatmap_fig = bx.get_figure()
    heatmap_fig.savefig(result_path+'/pics/hesse_{}.png'.format(epoch))
    plt.close()


def plot_loss(train_loss, valid_loss, train_cross_entropy, valid_cross_entropy):
    fig1 = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')
    # find position of lowest validation loss
    minposs = valid_cross_entropy.index(min(valid_cross_entropy)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig1.savefig(result_path+'/pics/{}_{}_loss.png'.format(CFG.dataset, CFG.model), bbox_inches='tight')
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_cross_entropy) + 1), train_cross_entropy, label='Training Cross Entropy')
    plt.plot(range(1, len(valid_cross_entropy) + 1), valid_cross_entropy, label='Validation Cross Entropy')
    # find position of lowest validation loss
    minposs = valid_cross_entropy.index(min(valid_cross_entropy)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('Cross Entropy')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_cross_entropy) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig2.savefig(result_path+'/pics/{}_{}_CrossEntropy.png'.format(CFG.dataset, CFG.model), bbox_inches='tight')
    plt.close(fig2)


if __name__ == '__main__':
    print(CFG)
    device = setup(CFG.device)
    train_loader, test_loader, valid_loader = getData(CFG.dataset, CFG.batch_size)
    net = getModel(CFG.model, device)
    optimizer = optim.Adam(net.parameters(), CFG.learning_rate)
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    train_loss, valid_loss, train_cross_entropy, valid_cross_entropy = train(net, CFG.epoch)
    test(net)
    plot_loss(train_loss, valid_loss, train_cross_entropy, valid_cross_entropy)
    torch.save(net.state_dict(), result_path+'/{}_{}_hesse.pth'.format(CFG.dataset, CFG.model))
