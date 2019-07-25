import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from ..inits import glorot, zeros


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        # 调用Parameter的__new__方法，必须将权重声明为Parameter,否则无法使用自动反向传播
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # 初始化参数
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    """edge_index加上自连接，邻接矩阵进行规范化"""
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            # edge_index.size(1)为有多少对边；edge_weight大小[edge_index.size(1)]
            edge_weight = torch.ones((edge_index.size(1), ), 
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) # 计算每个节点的度，维度为[num_nodes]
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        '''
        语法：row中元素的取值范围为[0,numnodes-1]，所以其值可以作为deg_inv_sqrt的下标索引
        Pytorch中 * 表示两个tensor对应位置相乘
        
        math:  \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}。其中\hat{D}}^{-1/2}为对角矩阵，
            对角矩阵左乘矩阵A，结果R1为 R1_{i,j}=D_{i,i}*A_{i,j}; A右乘对角矩阵，结果R2为 R2_{i,j}=A_{i,j}×D_{j,j}

        why: edge_weight与row维度均为有连接边的数目，采用这种方法比直接采用矩阵相乘复杂度降低，因为矩阵中通常会有很多0

        返回值：edge_index存储网络的连接结构(包括自连接)，维度为[2,edge_index中的边的数目]
        而后一项存储 \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}非零的元素，维度为[edge_index中的边的数目]
        后一项每个值与edge_index存在着对应关系。
        '''
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # 父类MessagePassing继承了torch.nn.Module，要想自定义层需要自定义forward函数，
    # 这里使用子类的forward函数间接的覆盖父类Module中的函数，backward使用Pytorch中的backward函数，可以自动求导
    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype) # x.size(0)为图的节点数目
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        # x为`\mathbf{X} \mathbf{\Theta}`
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """
        view(-1, 1)可以将维度[N,]转换为[N,1]；
        Pytorch中 * 表示两个tensor对应位置相乘，例如a维度为[N,1]，b维度[N,M]，则将a复制成[N,M]维度，然后对应位置相乘

        在原论文中，数学方法为`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta}`,
        为了解释方便，我们记\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}为矩阵A；\mathbf{X} \mathbf{\Theta}为矩阵B
        正常情况下若A和B均为标准矩阵，直接按照矩阵相乘法则即可，但是里面包含大量零元素，造成计算速度很慢。所以考虑稀疏矩阵的乘法。
        
        思想来源为：矩阵A右乘矩阵B相当于取A中的第i行对B中所有行进行线性组合，得到结果中的第i行。那么可以将矩阵A所有的非零元素抽出来，即这里的norm参数；
        并记录这些元素所在的行C1列C2；对于B按行取出来下标为C2的行向量组成新的矩阵，即这里的x_j。然后按照Pytorch中的tensor * 运算得到结果R；
        最后使用torch_scatter.scatter_add函数将R按照C1中相同值的下标按行相加，即完成了稀疏矩阵的乘法。

        注意：在MessagePassing.propagate方法`tmp = torch.index_select(tmp, 0, edge_index[idx])`中的tmp记为这里的实参x_j，而该语句中idx为0，
        是因为在cora数据集中，对于[1 2]链接的方向是从右到左的，所以这里的edge_index[0]即为我们上面提到的C2。同时也可以看到torch_scatter.scatter_add函数
        传参为edge_index[1]即为我们上面提到的C1。
        """
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    # 当打印该对象的时候，打印下面的话
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
