import sys
import os.path as osp
from itertools import repeat

import torch
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.read import read_txt_array
from torch_geometric.utils import remove_self_loops

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder, prefix):
    """
    - `x`, the feature vectors of the labeled training instances, for Cora:[140,1433]
    - `tx`, the feature vectors of the test instances, for Cora:[1000,1433]
    - `allx`, the feature vectors of both labeled and unlabeled training instances (a superset of `x`), for Cora:[1708,1433]
    - `y`, the one-hot labels of the labeled training instances, for Cora:[140,7]
    - `ty`, the one-hot labels of the test instances, for Cora:[1000,7]
    - `ally`, the labels for instances in `allx`. for Cora:[1708,7]
    - `graph`, a `dict` in the format `{index: [index_of_neighbor_nodes]}.` for Cora:[2708,*]
    - `test.index`, the indices of test instances in `graph`, for the inductive setting, for Cora:[1000]
    """
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long) # 从0到y.size(0)-1
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long) # 从y.size(0)到y.size(0)+499
    sorted_test_index = test_index.sort()[0] # sort方法 返回元组 (sorted_tensor, sorted_indices)

    if prefix.lower() == 'citeseer':
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = torch.zeros(len_test_indices, tx.size(1))
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = torch.zeros(len_test_indices, ty.size(1))
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    x = torch.cat([allx, tx], dim=0)
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1] # max方法 返回元组 (max_tensor, max_indices)

    x[test_index] = x[sorted_test_index]
    y[test_index] = y[sorted_test_index]

    train_mask = sample_mask(train_index, num_nodes=y.size(0))
    val_mask = sample_mask(val_index, num_nodes=y.size(0))
    test_mask = sample_mask(test_index, num_nodes=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y) # for Cora:[2,10556]
    data.train_mask = train_mask # 在类外添加属性
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0): # 获得Python的版本
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    # toarray returns an ndarray; todense returns a matrix
    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.Tensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value)) # 每一个key对应len(value)个值
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    # NOTE: There are duplicated edges and self loops in the datasets. Other
    # implementations do not remove them! 例如两个节点之间存在多条边
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index

# 产生[num_nodes]大小的全0tensor，index对应的位置置为1
def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask
