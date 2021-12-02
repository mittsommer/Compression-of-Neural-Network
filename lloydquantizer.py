import operator
from utils import *
import pandas as pd


def Quantization_with_lloyd_quantizer(e, module, dataset, batch_size, device, result_path, quantiz_level):
    result_levels = []
    for q_levels in quantiz_level:
        result_level = []
        net = getModel(module, device)
        # net = net.module
        net.load_state_dict(torch.load(result_path + '/epoch_{}.pth'.format(e), map_location=device))
        before_loss,before_acc = quantiz_test(net, dataset, batch_size, device)
        result_level.append(before_loss)
        result_level.append(before_acc)
        # ---------------quantization----------------------

        parm = net.parameters()
        for i, p in enumerate(parm):
            if i == 0:
                weights = p.data.view(-1, 1)
            else:
                weights = torch.cat((weights, p.data.view(-1, 1)))
        weights_numpy = torch.reshape(weights, (-1,)).cpu().numpy()
        weight_num = weights.data.__len__()
        minv = min(weights_numpy)
        maxv = max(weights_numpy)
        len = (-1 * minv + maxv) / q_levels
        m = np.zeros(q_levels + 1)
        for i in range(q_levels + 1):
            m[i] = minv + i * len
        sig = weights_numpy
        lu = np.zeros(q_levels)
        sk = np.zeros(q_levels)
        new = np.zeros(q_levels)
        Y = np.zeros(weight_num)
        for i in range(10):
            for k in range(q_levels):
                for j in range(weight_num):
                    if sig[j] < m[k+1] and sig[j] >= m[k]:
                        lu[k] = lu[k] + 1
                        sk[k] = sk[k] + sig[j]
                if lu[k] == 0:
                    sk[k] = (m[k] + m[k+1]) / 2
                    lu[k] = 1
                new[k] = sk[k] / lu[k]
                # print(new[k], k, 'new')
            for k in range(1, q_levels):
                m[k] = (new[k - 1] + new[k]) / 2
            for h in range(q_levels):
                for t in range(weight_num):
                    if weights_numpy[t] < m[h + 1] and weights_numpy[t] >= m[h]:
                        Y[t] = new[h]
        parm_ = net.named_parameters()
        temp = torch.from_numpy(Y)
        entropy = compute_entropy(temp).item()
        print('Entropy:', entropy)
        compression_rate = 32 / entropy
        print('Compression rate:', compression_rate)
        for name, param in parm_:
            netweight = operator.attrgetter(name)(net)
            size = 1
            for count in netweight.size():
                size = count * size
            temp = torch.split(temp, size)
            net_change = operator.attrgetter(name)(net)
            if device==torch.device('cuda:0'):
                net_change.data.copy_(nn.parameter.Parameter(temp[0].reshape(netweight.size()).type(torch.cuda.FloatTensor)))
            else:
                net_change.data.copy_(nn.parameter.Parameter(temp[0].reshape(netweight.size()).type(torch.FloatTensor)))
            if name != 'fc1.bias':
                temp = torch.cat(temp[1:], 0)
        # ------------------after quantization---------------------
        print('after quantization:')
        after_loss, after_acc = quantiz_test(net, dataset, batch_size, device)
        result_level.append(after_loss)
        result_level.append(after_acc)
        result_level.append(entropy)
        result_level.append(compression_rate)
        result_levels.append(result_level)
    result = pd.DataFrame(result_levels,
        columns=['loss_before', 'acc_before', 'loss_after', 'acc_after', 'entropy', 'compression_ratio'],
                          index=quantiz_level)
    result.to_csv(result_path+'/quantiz/lloydquantizer_{}.csv'.format(e))

