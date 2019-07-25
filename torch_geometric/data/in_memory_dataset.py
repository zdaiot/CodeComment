from itertools import repeat, product

import torch
from torch_geometric.data import Dataset, Data


class InMemoryDataset(Dataset):
    r"""Dataset base class for creating graph datasets which fit completely
    into memory.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder.""" # r是防止字符转义 
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(InMemoryDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter) # 初始化父类
        self.data, self.slices = None, None

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        data = self.data
        return data.y.max().item() + 1 if data.y.dim() == 1 else data.y.size(1)

    def __len__(self):
        return self.slices[list(self.slices.keys())[0]].size(0) - 1

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        Returns a data object, if :obj:`idx` is a scalar, and a new dataset in
        case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a LongTensor
        or a ByteTensor."""
        if isinstance(idx, int):
            data = self.get(idx)
            data = data if self.transform is None else self.transform(data) # 变换数据
            return data
        elif isinstance(idx, slice):
            return self.__indexing__(range(*idx.indices(len(self))))
        elif isinstance(idx, torch.LongTensor):
            return self.__indexing__(idx)
        elif isinstance(idx, torch.ByteTensor):
            return self.__indexing__(idx.nonzero())

        raise IndexError(
            'Only integers, slices (`:`) and long or byte tensors are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def shuffle(self):
        r"""Randomly shuffles the examples in the dataset."""
        return self.__indexing__(torch.randperm(len(self)))

    def get(self, idx):
        data = Data()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx].item()

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            # list切片，slice(None)相当于：；例如
            # 当item.dim()=2的时候，item[s]相当于[:,slices[idx], slices[idx + 1]]或者[slices[idx], slices[idx + 1],:]
            s = list(repeat(slice(None), item.dim())) # dim() 返回tensor的维数
            s[self.data.__cat_dim__(key, item)] = slice(
                slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    def __indexing__(self, index):
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__ = self.__dict__.copy()
        copy.data, copy.slices = self.collate([self.get(i) for i in index])
        return copy

    def collate(self, data_list):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        keys = data_list[0].keys # data_list: Data 类实例组成的list
        data = data_list[0].__class__() # 实例调用__class__() 返回类的一个新的实例
        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}
        # product(list1,list2)依次取出list1中每一个元素,与list2中的每一个元素组成元组,将所有元组组合成一个列表返回.
        for item, key in product(data_list, keys): 
            data[key].append(item[key]) # item[key] 自动调用 Data 类中的 __getitem__ 方法
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(
                    item.__cat_dim__(key, item[key]))
            elif isinstance(item[key], int) or isinstance(item[key], float):
                s = slices[key][-1] + 1
            else:
                raise ValueError('Unsupported attribute type.')
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)
            data.__num_nodes__ = torch.tensor(data.__num_nodes__)

        for key in keys:
            if torch.is_tensor(data_list[0][key]):
                # 若data_list包含多个Data实例，那么每一个data[key](list)均由所有Data实例中key对应元素组成，要进行拼接
                data[key] = torch.cat(
                    data[key], dim=data.__cat_dim__(key, data_list[0][key]))
            else:
                data[key] = torch.tensor(data[key])
            slices[key] = torch.tensor(slices[key], dtype=torch.long)
        # data为Data的实例，每一个key的值均由data_list中所有Data实例中key对应元素拼接而成

        # 考虑到data_list可能包含多个Data实例，而data是拼接出来的，所以需要slices存储每一个data[key]中不同Data实例的范围
        # 例如[0,2708]，表示一个data实例，范围为从0到2708。
        return data, slices
