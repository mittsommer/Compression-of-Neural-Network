import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models import LeNet3, LeNet5
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def setup(d):
    """SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)"""
    device = torch.device(d)
    return device


def getData(name, batch_size):
    transform = transforms.ToTensor()
    if name == 'cifar10':
        train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    if name == 'MNIST':
        train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
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
                                               num_workers=0)
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)
    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=0)
    return train_loader, test_loader, valid_loader



def getModel(name, device):
    if name == "LeNet3":
        net = LeNet3().to(device)
    if name == "LeNet5":
        net = LeNet5().to(device)
    return net


def eval_hessian(first_grad, model, device):
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    cnt = 0
    for g in first_grad:
        g_vector = g.view(-1) if cnt == 0 else torch.cat([g_vector, g.view(-1)])
        cnt = 1
    weights_number = g_vector.size(0)
    hessian_matrix = torch.zeros(weights_number, weights_number).to(device)
    for idx in range(weights_number):
        second_grad = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True, retain_graph=True)
        cnt = 0
        for g in second_grad:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian_matrix[idx] = g2
    return hessian_matrix


'''def save_log():
    log_csv = pd.read_csv('./log.csv')
    log = pd.DataFrame(
        [[CFG['batch_size'], CFG['learning_rate'], CFG['epoch'], CFG['lambda'], round(test_loss, 6),
          round(crossentropy, 6), round(curvature, 6), accuracy]],
        columns=['batch_size', 'learning_rate', 'epoch', 'lambda', 'test_loss', 'crossentropy', 'curvature',
                 'accuracy'])
    log_csv = log_csv.append(log, ignore_index=True)
    log_csv.to_csv('log.csv', index=False)'''
