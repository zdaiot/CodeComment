import torch

from .num_nodes import maybe_num_nodes


def contains_self_loops(edge_index):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` does not
    contain self-loops.

    Args:
        edge_index (LongTensor): The edge indices.

    :rtype: bool
    """
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col # 理解方法 mask = (row != col)
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, all existent self-loops will be removed and
    replaced by weights denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0,
                              num_nodes,
                              dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains a few self-loops, only
    non-existent self-loops will be added with weights denoted by
    :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index # 将edge_index第一行赋值给row，第二行赋值给col

    mask = row != col # 理解方法 mask = (row != col)
    inv_mask = 1 - mask # mask中1变为0，而0变成1，相当于 inv_mask = (row == col)
    # 生成大小为[numnodes]的tensor，并使用fill_value值进行填充
    loop_weight = torch.full(
        (num_nodes, ),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        # numel()返回一个tensor变量内所有元素个数
        assert edge_weight.numel() == edge_index.size(1)
        # 该语句的作用为若原本的edge_index存在自连接，则直接使用原本的自连接权重
        loop_weight[row[inv_mask]] = edge_weight[inv_mask].view(-1)
        # 这里使用[mask]是为了剔除掉edge_weight中的权重，因为loop_weight中已经包含了所有自连接边的权重了
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(0,
                              num_nodes,
                              dtype=torch.long,
                              device=row.device)
    # unsqueeze增加一维，例如[...]变成[[...]]，repeat(2, 1)后变成[[...],[...]]；这里使用该方法表示节点自连接的起点和终点
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    # 这里使用[:, mask]，是为了防止原本edge_index中含有自连接结构，因为loop_index中已经包含了所有的自连接边结构了
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
    # 返回带有自连接结构的edge_index，和带有自连接权重的edge_weight，两者之间一一对应；且自连接部分均在两者的后端
    return edge_index, edge_weight
