import numpy as np
import scipy.sparse as sp
import torch
import os,sys

'''输入大小为(N,)的numpy数组'''
def encode_onehot(labels):
    classes = set(labels)
    # np.identity生成单位矩阵，对于每一个类别C都对应一个one-hot向量。例如{one:[1,0,0,0,0,0,0],...}
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # dict.get(key, default=None)返回指定键的值；这里使用map用法将每一个label扔到get方法中，得到该类标对应的one-hot向量
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"): # path=os.path.join(sys.path[0],"../data/cora/")
    """Load citation network dataset (cora only for now)""" 
    print('Loading {} dataset...'.format(dataset))
    # 将 .content 文件以 numpy的形式读入，其中每一个元素的类型均为np.dtype(str)
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # 1:-1去掉第一列和最后一列，csr_matrix行压缩矩阵
    labels = encode_onehot(idx_features_labels[:, -1]) # [:, -1]最后一列为每个元素的类标

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32) # [:, 0]第一列为文章的编号
    idx_map = {j: i for i, j in enumerate(idx)} # j编号文章在文件中的第i行
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape) # 将cites中文章的编号映射到.content中的真实行
    # 生成(edges.shape[0],)大小的矩阵，并将第i个值放置到矩阵的edges[:, 0](i), edges[:, 1](i)位置
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix，生成对称邻接矩阵
    # multiply：对应元素相乘；若对称位置元素值不相同，则取较大的值
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # 邻接矩阵加上自连接并求得 \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense())) # todense返回matrix
    labels = torch.LongTensor(np.where(labels)[1]) # np.where当输入为二维数组时，返回两个list，第一个list存放行，第二个list存放列
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    # 对于邻接矩阵，相当于求 \hat{D}
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # 矩阵左乘对角矩阵相当于对角矩阵的每一个元素分别乘以对应行
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) # max(1)计算出数据中一行的最大值，并输出最大值所在的列号
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # tocoo：把矩阵转换成坐标形式，可以有效地存储和处理大多数元素为零的张量。
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
