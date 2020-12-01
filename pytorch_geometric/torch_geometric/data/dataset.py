import collections
import os.path as osp

import torch.utils.data

from .makedirs import makedirs


def to_list(x):
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x


def files_exist(files):
    return all([osp.exists(f) for f in files])


class Dataset(torch.utils.data.Dataset):
    r"""Dataset base class for creating graph datasets.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_dataset.html>`__ for the accompanying tutorial.

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
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __len__(self):
        r"""The number of examples in the dataset."""
        raise NotImplementedError

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(Dataset, self).__init__()
        """print(self) Cora()"""
        # expanduser 把path中包含的"~"和"~user"转换成用户目录
        # normpath规范化路径 在Linux和Mac平台上，该函数会原样返回path，在windows平台上会将路径中所有字符转换为小写，并将所有斜杠转换为饭斜杠。
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'raw')
        self.processed_dir = osp.join(self.root, 'processed')
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self._download() # 判断是否存在数据，若不存在则下载数据
        self._process()

    @property
    def num_features(self):
        r"""Returns the number of features per node in the graph."""
        return self[0].num_features

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        files = to_list(self.raw_file_names)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = to_list(self.processed_file_names) # self.processed_file_names 在子类Planetoid中覆盖
        return [osp.join(self.processed_dir, f) for f in files]

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        # 此时调用的是Planetoid中的download，因为子类的方法将父类的方法覆盖了。

        """可以理解为当通过子类的super().__init__方法调用父类初始化函数的时候，子类继承并覆盖了父类的属性：
        执行父类构造函数，向子类中添加子类中未定义的属性；对于子类和父类均定义的方法和属性，采用的是子类中方法和属性；
        所以执行父类中的方法，若子类已经覆盖，则执行子类中的方法。"""
        self.download() # <bound method Planetoid.download of Cora()>

    def _process(self):
        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print('Processing...')

        makedirs(self.processed_dir)
        self.process() # 调用Planetoid子类中的process方法

        print('Done!')

    def __getitem__(self, idx):  # pragma: no cover
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given)."""
        data = self.get(idx)
        data = data if self.transform is None else self.transform(data)
        return data

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))
