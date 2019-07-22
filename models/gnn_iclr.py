#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Pytorch requirements
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor


def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)  # 在维度3上进行分割，每个小组的组块大小为1。最终数据的维度为[bs, N, N, 1]_{J}
    # torch.cat(W, 1)将数据沿着第二个维度拼接，得到的大小为[bs,N*J,N,1]，squeeze(3)去掉第四个维度，W is now a tensor of size (bs, J*N, N)
    # 相当于将[N, N, J]三维数据，转为了[N*J,N]的二维数据
    W = torch.cat(W, 1).squeeze(3)
    output = torch.bmm(W, x)  # bmm为批矩阵相乘，要求输入尺寸为三维tensor，output has size (bs, J*N, num_features)
    output = output.split(N, 1)  # output维度为[bs, N, num_features]_{J}
    output = torch.cat(output, 2)  # output has size (bs, N, J*num_features)
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J * nf_input # 经过J操作后，数据输入维度的大小
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    # input为[W,x]
    def forward(self, input):
        W = input[0]
        x = gmul(input)  # out has size (bs, N, num_inputs) J*num_features=num_inputs
        # if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x)  # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs) # has size [bs, N , num_outputs]
        return W, x


class Wcompute(nn.Module):
    '''
    input_features：输入特征的维度
    nf：ratio中4个比例的基数
    ratio：分别为4个卷积层的出输出尺寸比例
    '''

    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1], num_operators=1,
                 drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf * ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf * ratio[2])
        self.conv2d_4 = nn.Conv2d(nf * ratio[2], nf * ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf * ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    # 输入x的维度为[batch_size,n_way*num_shots+1,emb_size+n_way]
    # 输入W_id维度为[batch_size,n_way*num_shots+1,n_way*num_shots+1,1]
    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)  # 维度[batch_size,n_way*num_shots+1,1,emb_size+n_way]
        W2 = torch.transpose(W1, 1, 2)  # 维度为[batch_size,1,n_way*num_shots+1,emb_size+n_way]
        # Pytorch中两个tensor相减，若维度不同，则先扩张到相同尺度，然后相减
        W_new = torch.abs(W1 - W2)  # 维度为[batch_size,n_way*num_shots+1,n_way*num_shots+1,emb_size+n_way]
        # 维度为[batch_size,emb_size+n_way,n_way*num_shots+1,n_way*num_shots+1]，转换维度是因为conv2d输入数据的第二个维度为channel
        W_new = torch.transpose(W_new, 1, 3)

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3)  # 维度[batch_size,n_way*num_shots+1,n_way*num_shots+1,1]

        # 激活函数
        if self.activation == 'softmax':
            '''
            主要介绍一下使用softmax对每一行完成归一化操作
            
            W_new为网络学习到的拓扑结构，W_id始终为单位矩阵
            将W_new-W_id乘以*1e8，是为了经过softmax之后对角线元素接近0；即忽略自身与自身的连接

            思路为：首先将数据从维度[batch_size,n_way*num_shots+1,n_way*num_shots+1,1]转为[batch_size,n_way*num_shots+1,1,n_way*num_shots+1]，
            此时最后一个维度即为每行数据，然后view成尺寸[batch_size*n_way*num_shots+1,n_way*num_shots+1]，此时即可使用softmax函数；
            最后将softmax后的数据，转换成尺寸[batch_size,n_way*num_shots+1,n_way*num_shots+1,1]
            '''
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            # tensor转置后共享一块内存，只是修改了一些属性。使用contiguous方法可以使得当前tensor布局与直接声明该尺寸的tensor布局一致
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)
        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)  # tensor中所有元素均经过sigmoid函数
            W_new *= (1 - W_id)  # 对角线为0，其余为为1；即忽略自身与自身的连接
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        # 为'J2'的时候，we are using two operators, one is the identity matrix, and the other one is defined by the adjacency matrix of the graph.
        # 最终得到的W_news的维度为[batch_size,n_way*num_shots+1,n_way*num_shots+1,2]
        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise (NotImplementedError)

        return W_new


''' 适用于omniglot数据集在few-shot learning以及半监督的GNN
input_features：输入数据的维度
nf： Number of Features for each convolutional layer，这里为96
J："J" is the number of operators of the Graph Neural Network. 
    In our case J=2, since we are using two operators, one is the identity matrix, and the other one is defined by the adjacency matrix of the graph.
    这里为1，因为此参数没有作用，其作用被operator='J2'取代
'''
class GNN_nl_omniglot(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl_omniglot, self).__init__()
        self.args = args
        self.input_features = input_features  # 输入数据的尺度
        self.nf = nf
        self.J = J

        self.num_layers = 2
        for i in range(self.num_layers):
            module_w = Wcompute(self.input_features + int(nf / 2) * i,
                                self.input_features + int(nf / 2) * i,
                                operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=False)
            module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            # add_module(name,module) 将子模块加入当前的模块中，被添加的模块可以self._modules[name]来获取
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers,
                                    self.input_features + int(self.nf / 2) * (self.num_layers - 1),
                                    operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2,
                                bn_bool=True)

    def forward(self, x):
        '''
        GNN的结构为：Wcompute+GCN+ Wcompute+GCN+ Wcompute+GCN

        输入x的维度为[batch_size,n_way*num_shots+1,emb_size+n_way]
        torch.eye产生大小为[x.size(1),x.size(1)]的单位矩阵，最终W_init维度为[batch_size,x.size(1),x.size(1),1]
        '''
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)  # 调用Wcompute的forword函数

            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1] # has size [batch_size, n_way*num_shots+1, n_way]

        return out[:, 0, :] # has size [batch_size, n_way]，这里之所以取第二个维度0，是因为在输入x中batch_size组数据中，每组的第一个数据均为测试样本


class GNN_nl(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        if args.dataset == 'mini_imagenet':
            self.num_layers = 2
        else:
            self.num_layers = 2

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax',
                                    ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2',
                                    activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2,
                                bn_bool=False)

    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)

            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]


class GNN_active(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_active, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 2
        for i in range(self.num_layers // 2):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax',
                                    ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)

            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.conv_active_1 = nn.Conv1d(self.input_features + int(nf / 2) * 1, self.input_features + int(nf / 2) * 1, 1)
        self.bn_active = nn.BatchNorm1d(self.input_features + int(nf / 2) * 1)
        self.conv_active_2 = nn.Conv1d(self.input_features + int(nf / 2) * 1, 1, 1)

        for i in range(int(self.num_layers / 2), self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax',
                                    ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2',
                                    activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2,
                                bn_bool=False)

    def active(self, x, oracles_yi, hidden_labels):
        x_active = torch.transpose(x, 1, 2)
        x_active = self.conv_active_1(x_active)
        x_active = F.leaky_relu(self.bn_active(x_active))
        x_active = self.conv_active_2(x_active)
        x_active = torch.transpose(x_active, 1, 2)

        x_active = x_active.squeeze(-1)
        x_active = x_active - (1 - hidden_labels) * 1e8
        x_active = F.softmax(x_active)
        x_active = x_active * hidden_labels

        if self.args.active_random == 1:
            # print('random active')
            x_active.data.fill_(1. / x_active.size(1))
            decision = torch.multinomial(x_active, 1)
            x_active = x_active.detach()
        else:
            if self.training:
                decision = torch.multinomial(x_active, 1)
            else:
                _, decision = torch.max(x_active, 1)
                decision = decision.unsqueeze(-1)

        decision = decision.detach()

        mapping = torch.FloatTensor(decision.size(0), x_active.size(1)).zero_()
        mapping = Variable(mapping)
        if self.args.cuda:
            mapping = mapping.cuda()
        mapping.scatter_(1, decision, 1)

        mapping_bp = (x_active * mapping).unsqueeze(-1)
        mapping_bp = mapping_bp.expand_as(oracles_yi)

        label2add = mapping_bp * oracles_yi  # bsxNodesxN_way
        padd = torch.zeros(x.size(0), x.size(1), x.size(2) - label2add.size(2))
        padd = Variable(padd).detach()
        if self.args.cuda:
            padd = padd.cuda()
        label2add = torch.cat([label2add, padd], 2)

        x = x + label2add
        return x

    def forward(self, x, oracles_yi, hidden_labels):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers // 2):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        x = self.active(x, oracles_yi, hidden_labels)

        for i in range(int(self.num_layers / 2), self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]


if __name__ == '__main__':
    # test modules
    bs = 4
    nf = 10
    num_layers = 5
    N = 8
    x = torch.ones((bs, N, nf))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### test gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out[0, :, num_features:])
    ######################### test gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### test gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())
