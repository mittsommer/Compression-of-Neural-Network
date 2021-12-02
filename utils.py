import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models import *
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
import math


def set_device(gpu=True):
    """SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)"""
    if gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def getData(name, batch_size):
    transform = transforms.ToTensor()
    if name == 'cifar10':
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    if name == 'MNIST':
        train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    if name == 'FashionMNIST':
        train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    # percentage of training set to use as validation
    valid_size = 0.2
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # load training data in batches
    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=1)
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=1)
    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=1)
    return train_loader, test_loader, valid_loader


def getModel(name, device):
    if name == "LeNet3":
        net = LeNet3().to(device)
    if name == "LeNet3_2":
        net = LeNet3_2().to(device)
    if name == "LeNet3_3":
        net = LeNet3_3().to(device)
    if name == "LeNet3_4":
        net = LeNet3_4().to(device)
    if name == "LeNet_5":
        net = LeNet_5().to(device)
    if name == "alexnet":
        net = AlexNet().to(device)

    return net


def quantiz_test(net, dataset, batch_size, device):
    net.eval()
    train_loader, test_loader, valid_loader = getData(dataset, batch_size)
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data).to(device)
        t = criterion(output, target)
        test_loss += t.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset),
                                                                              accuracy))
    return test_loss, accuracy


def hessian(first_grad, net, device):
    cnt = 0
    for fg in first_grad:
        if cnt == 0:
            first_vector = fg.contiguous().view(-1)
            cnt = 1
        else:
            first_vector = torch.cat([first_vector, fg.contiguous().view(-1)])
    weights_number = first_vector.size(0)
    hessian_matrix = torch.zeros(weights_number, weights_number).to(device)
    for idx in range(weights_number):
        second_grad = torch.autograd.grad(first_vector[idx], net.parameters(), create_graph=True, retain_graph=True)
        cnt = 0
        for g in second_grad:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian_matrix[idx] = g2
    return hessian_matrix


def hessian_topk(first_grad, net, k):
    cnt = 0
    for fg in first_grad:
        first_vector = fg.view(-1) if cnt == 0 else torch.cat([first_vector, fg.view(-1)])
        cnt = 1
    weights_number = first_vector.size(0)
    curvature = []
    for idx in range(weights_number):
        second_grad = torch.autograd.grad(first_vector[idx], net.parameters(), create_graph=True, retain_graph=True)
        cnt = 0
        for sg in second_grad:
            second_vector = sg.contiguous().view(-1) if cnt == 0 else torch.cat(
                [second_vector, sg.contiguous().view(-1)])
            cnt = 1
        second_topk = torch.topk(second_vector, k=int((weights_number - idx) * k + 1))[0]

        curvature.append(torch.pow(second_topk, 2))
        '''curvature = torch.sum(torch.pow(second_topk, 2)) if idx == 0 else curvature + torch.sum(
            torch.pow(second_topk, 2))'''

        # print('{}/{}'.format(idx,weights_number))
    return torch.sum(torch.tensor(curvature))


def hessian_random_topk(first_grad, net, k):
    cnt = 0
    for g in first_grad:
        first_vector = g.view(-1) if cnt == 0 else torch.cat([first_vector, g.view(-1)])
        cnt = 1
    weights_number = first_vector.size(0)
    random_idx = torch.randint(0, weights_number - 1, (int(0.001 * weights_number),))
    for idx in random_idx:
        second_grad = torch.autograd.grad(first_vector[idx], net.parameters(), create_graph=True, retain_graph=True)
        cnt = 0
        for g in second_grad:
            second_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([second_vector, g.contiguous().view(-1)])
            cnt = 1
        curvature = torch.sum(torch.pow(torch.topk(second_vector, k=int((weights_number - idx) * k + 1))[0], 2)) \
            if idx == random_idx[0] else curvature + torch.sum(
            torch.pow(torch.topk(second_vector[idx:], k=int((weights_number - idx) * k + 1))[0], 2))
        # print('{}/{}'.format(idx,weights_number))
    return curvature


def hessian_diagonal(first_grad, net,device):
    second_grad = []
    for i, parm in enumerate(net.parameters()):
        second_grad.append(torch.autograd.grad(first_grad[i], parm, retain_graph=True, create_graph=True,
                                               grad_outputs=torch.ones_like(first_grad[i]))[0])
    curvature = torch.tensor(0)
    for i in second_grad:
        curvature = curvature + torch.sum(torch.pow(i, 2))
    return curvature


'''def save_log():
    log_csv = pd.read_csv('./log.csv')
    log = pd.DataFrame(
        [[CFG['batch_size'], CFG['learning_rate'], CFG['epoch'], CFG['lambda'], round(test_loss, 6),
          round(crossentropy, 6), round(curvature, 6), accuracy]],
        columns=['batch_size', 'learning_rate', 'epoch', 'lambda', 'test_loss', 'crossentropy', 'curvature',
                 'accuracy'])
    log_csv = log_csv.append(log, ignore_index=True)
    log_csv.to_csv('log.csv', index=False)'''


def compute_entropy(labels, base=None):
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = torch.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = torch.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = 2 if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent


def pdf(bundies, weights):
    cluster = torch.zeros_like(weights)
    prob = torch.zeros(len(bundies) - 1)
    for w in range(len(weights)):
        weight = weights[w]
        for b in range(len(bundies) - 1):
            if bundies[b] <= weight <= bundies[b + 1]:
                prob[b] += 1
                cluster[w] = b
    prob = prob / len(weights)
    return cluster, prob


def plot_hesse(hesse, epoch, result_path):
    ax = sns.distplot((hesse.flatten()).cpu().detach().numpy(), color='b', kde=False, bins=20, hist_kws={'log': True})
    ax.set_xlabel('Values of second derivatives')
    ax.set_ylabel('frequences')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(10, 400000)
    dist_fig = ax.get_figure()
    dist_fig.savefig(result_path + '/hesse_distribution_{}.png'.format(epoch))
    plt.close()
    bx = sns.heatmap(hesse.cpu().detach().numpy(), vmin=-0.5, vmax=0.5)
    bx.set_xlabel('row')
    bx.set_ylabel('column')
    plt.title('Hesse Matrix')
    ax.xaxis.tick_top()
    heatmap_fig = bx.get_figure()
    heatmap_fig.savefig(result_path + '/hesse_{}.png'.format(epoch))
    plt.close()


def plot_loss(train_loss, valid_loss, train_cross_entropy, valid_cross_entropy, result_path):
    fig1 = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')
    # find position of lowest validation loss
    '''minposs = valid_cross_entropy.index(min(valid_cross_entropy)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')'''
    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig1.savefig(result_path + '/loss.png', bbox_inches='tight')
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_cross_entropy) + 1), train_cross_entropy, label='Training Cross Entropy')
    plt.plot(range(1, len(valid_cross_entropy) + 1), valid_cross_entropy, label='Validation Cross Entropy')
    # find position of lowest validation loss
    '''minposs = valid_cross_entropy.index(min(valid_cross_entropy)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')'''
    plt.xlabel('epochs')
    plt.ylabel('Cross Entropy')
    #plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_cross_entropy) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig2.savefig(result_path + '/CrossEntropy.png', bbox_inches='tight')
    plt.close(fig2)


def plot_weights(model,result_path):
    modules = [module for module in model.modules()]
    num_sub_plot = 0
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            plt.subplot(131 + num_sub_plot)
            w = layer.weight.data
            w_one_dim = w.cpu().numpy().flatten()
            plt.hist(w_one_dim[w_one_dim != 0], bins=50)
            num_sub_plot += 1
    plt.savefig(result_path)


def plot_all_weights(model, result_path):
    modules = [module for module in model.modules()]
    modules = modules[1:]
    all_weights = torch.tensor([])
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            w = layer.weight.data.view(-1).cpu()
            all_weights = torch.cat((all_weights, w))
    sns.histplot(all_weights.cpu().numpy(),bins=50)
    plt.savefig(result_path, bbox_inches='tight')
    plt.close()


