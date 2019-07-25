import torch_scatter


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']
    # 获得torch_scatter中的'scatter_{}'.format(name)属性，这里得到的op为函数
    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e9 if name == 'max' else 0
    """
    以torch_scatter.scatter_add函数为例，若index为[0,0,1,0,2,2,3,3]，而src为[5,1,7,2,3,2,1,3]，
    则将index中值0对应在src位置的值相加得到第一个值，这里index值为0的位置为[0,1,3]，在src里这些位置的值相加5+1+2=8。
    依次操作，直到取到index的最大值。
    
    可以在 https://rusty1s.github.io/pytorch_scatter/build/html/functions/add.html 看到更加详细的文档
    """
    out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out
