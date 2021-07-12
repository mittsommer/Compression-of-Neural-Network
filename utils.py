import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models import LeNet3, LeNet5
import torchvision


def setup(d):
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device(d)
    return device


def getData(name, batch_size):
    transformer = transforms.Compose([torchvision.transforms.ToTensor()])
    if name == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transformer)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    if name == 'MNIST':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transformer)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = datasets.MNIST(root='./data', train=False, download=False, transform=transformer)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def getModel(name, device):
    if name == "LeNet3":
        net = LeNet3().to(device)
    if name == "LeNet5":
        net = LeNet5().to(device)
    return net


def eval_hessian(first_grad, model):
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    cnt = 0
    for g in first_grad:
        g_vector = g.view(-1) if cnt == 0 else torch.cat([g_vector, g.view(-1)])
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


'''def save_log():
    log_csv = pd.read_csv('./log.csv')
    log = pd.DataFrame(
        [[CFG['batch_size'], CFG['learning_rate'], CFG['epoch'], CFG['lambda'], round(test_loss, 6),
          round(crossentropy, 6), round(curvature, 6), accuracy]],
        columns=['batch_size', 'learning_rate', 'epoch', 'lambda', 'test_loss', 'crossentropy', 'curvature',
                 'accuracy'])
    log_csv = log_csv.append(log, ignore_index=True)
    log_csv.to_csv('log.csv', index=False)'''