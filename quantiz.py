import argparse
from lloydquantizer import *
from uniform_quantizer import *
from utils import *
import sys
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--l", default=0.99, help="lambda", type=float)
parser.add_argument("--dataset", default='FashionMNIST')
parser.add_argument("--model", default='LeNet3_3')
parser.add_argument("--quantiz_level", default=[256, 128, 64, 32, 16, 12, 10, 8, 6, 4, 2])
parser.add_argument("--time", default=1, type=int)
parser.add_argument("--gpu", action='store_false', default=True)
CFG = parser.parse_args()

result_path = './result/{}_{}/time{}/lambda_{}'.format(CFG.model, CFG.dataset, CFG.time, CFG.l)
if not os.path.exists(result_path + '/quantiz'):
    os.makedirs(result_path + '/quantiz')


def test(net, dataset, batch_size, device):
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


class Logger(object):
    def __init__(self, filename=result_path + '/quantiz.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    sys.stdout = Logger(stream=sys.stdout)
    print(CFG)
    print('########The model is:', CFG.model, '########')
    device = set_device(CFG.gpu)
    print('########lambda is:', CFG.l, '########')
    '''for e in range(1, CFG.epoch + 1):
        print('############ epoch: {} ############'.format(e))
        print('############ lloyd_quantizer ############')
        Quantization_with_lloyd_quantizer(e, CFG.model, CFG.dataset, CFG.batch_size, device, result_path,
                                          CFG.quantiz_level)
        print('############ uniform_quantizer ############')
        Quantization_with_uniform_quantizer(e, CFG.model, CFG.dataset, CFG.batch_size, device, result_path,
                                            CFG.quantiz_level)'''
    e = CFG.epoch
    print('############ epoch: {} ############'.format(e))
    print('############ lloyd_quantizer ############')
    Quantization_with_lloyd_quantizer(e, CFG.model, CFG.dataset, CFG.batch_size, device, result_path,
                                      CFG.quantiz_level)
    print('############ uniform_quantizer ############')
    Quantization_with_uniform_quantizer(e, CFG.model, CFG.dataset, CFG.batch_size, device, result_path,
                                        CFG.quantiz_level)
