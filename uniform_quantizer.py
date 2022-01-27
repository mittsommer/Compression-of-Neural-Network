import operator
from sklearn.metrics import mean_squared_error
from utils import *
import pandas as pd


def Quantization_with_uniform_quantizer(e, module, dataset, batch_size, device, result_path, quantiz_level):
    result_levels = []
    for q_levels in quantiz_level:
        result_level = []
        net = getModel(module, device)
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
        NGEN = 5
        popsize = 20
        low = 0.0001
        up = 3 / q_levels
        parameters = [NGEN, popsize, low, up, q_levels, weights_numpy]
        aco = ACO(parameters)
        footstep = aco.main()
        print('The foot step is:', footstep)

        m = np.zeros(q_levels + 1)
        meanv = np.mean(weights_numpy)
        for i in range(int(q_levels / 2 + 1)):
            m[int(q_levels / 2 + i)] = meanv + i * footstep
            m[int(q_levels / 2 - i)] = meanv - i * footstep
        Y = np.zeros(weights_numpy.size)
        for h in range(q_levels):
            for t in range(weights_numpy.size):
                if weights_numpy[t] < m[h + 1] and weights_numpy[t] >= m[h]:
                    Y[t] = (m[h] + m[h + 1]) / 2
        for t in range(weights_numpy.size):
            if weights_numpy[t] < m[0]:
                Y[t] = m[0]
            if weights_numpy[t] > m[q_levels]:
                Y[t] = m[q_levels]

        parm_ = net.named_parameters()
        temp = torch.from_numpy(Y)
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
        # ------------------after kmeans---------------------
        print('after quantization:')
        after_loss, after_acc = quantiz_test(net, dataset, batch_size, device)
        result_level.append(after_loss)
        result_level.append(after_acc)
        result_levels.append(result_level)
    result = pd.DataFrame(result_levels,
                          columns=['loss_before', 'acc_before', 'loss_after', 'acc_after'],
                          index=quantiz_level)
    result.to_csv(result_path + '/quantiz/uniformdquantizer_{}.csv'.format(e))


class ACO:
    def __init__(self, parameters):
        """
        Ant Colony Optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        self.NGEN = parameters[0]
        self.pop_size = parameters[1]
        self.var_num = 1
        self.bound = []
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
        self.qlevels = parameters[4]
        self.weights = parameters[5]

        self.pop_x = np.zeros(self.pop_size)
        self.g_best = np.zeros(1)

        for i in range(self.pop_size):
            self.pop_x[i] = np.random.uniform(self.bound[0], self.bound[1])
            fit = self.fitness(self.pop_x[i])
            if i == 0:
                temp = fit
            else:
                if fit < temp:
                    self.g_best = self.pop_x[i]
                    temp = fit

    def fitness(self, pop_x):
        m = np.zeros(self.qlevels + 1)
        meanv = sum(self.weights) / self.weights.size
        for i in range(int(self.qlevels / 2 + 1)):
            m[int(self.qlevels / 2 + i)] = meanv + i * pop_x
            m[int(self.qlevels / 2 - i)] = meanv - i * pop_x
        Y = np.zeros(self.weights.size)
        for h in range(self.qlevels):
            for t in range(self.weights.size):
                if self.weights[t] < m[h + 1] and self.weights[t] >= m[h]:
                    Y[t] = (m[h] + m[h + 1]) / 2
        for t in range(self.weights.size):
            if self.weights[t] < m[0]:
                Y[t] = m[0]
            if self.weights[t] > m[self.qlevels]:
                Y[t] = m[self.qlevels]
        mseloss = mean_squared_error(Y, self.weights)
        return mseloss

    def update_operator(self, gen, t, t_max):
        rou = 0.8
        Q = 1
        lamda = 1 / gen
        pi = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            pi[i] = (t_max - t[i]) / t_max
            if pi[i] < np.random.uniform(0, 1):
                self.pop_x[i] = self.pop_x[i] + np.random.uniform(-1, 1) * lamda
            else:
                self.pop_x[i] = self.pop_x[i] + np.random.uniform(-1, 1) * (
                        self.bound[1] - self.bound[0]) / 2
            if self.pop_x[i] < self.bound[0]:
                self.pop_x[i] = self.bound[0]
            if self.pop_x[i] > self.bound[1]:
                self.pop_x[i] = self.bound[1]
            t[i] = (1 - rou) * t[i] + Q * self.fitness(self.pop_x[i])
            if self.fitness(self.pop_x[i]) < self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
        t_max = np.max(t)
        return t_max, t

    def main(self):
        popobj = []
        best = np.zeros(1)
        for gen in range(1, self.NGEN + 1):
            if gen == 1:
                tmax, t = self.update_operator(gen, np.array(list(map(self.fitness, self.pop_x))),
                                               np.max(np.array(list(map(self.fitness, self.pop_x)))))
            else:
                tmax, t = self.update_operator(gen, t, tmax)
            popobj.append(self.fitness(self.g_best))
            # print('############ Generation {} ############'.format(str(gen)))
            # print(self.g_best)
            # print(self.fitness(self.g_best))
            if self.fitness(self.g_best) < self.fitness(best):
                best = self.g_best.copy()
        #     print('Best step size：{}'.format(best))
        #     print('Min loss：{}'.format(self.fitness(best)))
        # print("---- End of (successful) Searching ----")

        return best
