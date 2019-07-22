from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import random
from torch.autograd import Variable
from . import omniglot
from . import mini_imagenet


class Generator(data.Dataset):
    def __init__(self, root, args, partition='train', dataset='omniglot'):
        self.root = root
        self.partition = partition  # training set or test set
        self.args = args

        assert (dataset == 'omniglot' or
                dataset == 'mini_imagenet'), 'Incorrect dataset partition'
        self.dataset = dataset
        # 输入数据尺寸
        if self.dataset == 'omniglot':
            self.input_channels = 1 # 灰度图
            self.size = (28, 28)
        else:
            self.input_channels = 3
            self.size = (84, 84)

        if dataset == 'omniglot':
            self.loader = omniglot.Omniglot(self.root, dataset=dataset)
            self.data = self.loader.load_dataset(self.partition == 'train', self.size)
        elif dataset == 'mini_imagenet':
            self.loader = mini_imagenet.MiniImagenet(self.root)
            self.data, self.label_encoder = self.loader.load_dataset(self.partition, self.size)
        else:
            raise NotImplementedError

        self.class_encoder = {} # 存放类标的映射关系，例如{0: 0, 1: 1}
        for id_key, key in enumerate(self.data):
            self.class_encoder[key] = id_key

    def rotate_image(self, image, times):
        rotated_image = np.zeros(image.shape)
        for channel in range(image.shape[0]):
            rotated_image[channel, :, :] = np.rot90(image[channel, :, :], k=times)
        return rotated_image
    
    '''取数据，n_ways参数表示有每次取多少类，num_shots表示每一类有多少个数据

    这里对额外无类标数据(用于半监督学习)的处理方式为:取n_ways个类别，每个类别取出num_shot个数据，然后去掉其中unlabeled_extra个数据的类标
    例如，num_shot=5，unlabeled_extra=4，相当于每类有1个有类标数据，4个无类标数据
    '''
    def get_task_batch(self, batch_size=5, n_way=20, num_shots=1, unlabeled_extra=0, cuda=False, variable=False):
        '''Init variables
        batch_x存放测试集的数据，维度为[batch_size, self.input_channels, self.size[0], self.size[1]]
        label_x存放测试集的类标(该类标为被选中的类在n_ways类中的序号),维度为[batch_size, n_way]
        labels_x_global存放测试样本在所有类中的序号，大小为batch_size

        target_distances存放     ，维度为[batch_size, n_way * num_shots]
        hidden_labels存放    ，维度为[batch_size, n_way * num_shots + 1]，在active learning中使用
        numeric_labels存放每次迭代测试样本所在类在n_ways类中的序号，大小为batch_size

        batches_xi存放训练集(有类标和额外无类标)的数据，维度为[batch_size, self.input_channels, self.size[0], self.size[1]]_{n_way*num_shots}
        labels_yi存放训练集(有类标)的类标，维度为[batch_size, n_way]_{n_way*num_shots}
        oracles_yi存放训练集(有类标和额外无类标)的类标，维度为[batch_size, n_way]_{n_way*num_shots}，在active learning中使用

        可以看出labels_yi和oracles_yi的维度相同，区别在于在半监督学习中，对于num_shots中的前unlabeled_extra个数据，
        对应在labels_yi中为全零向量(与训练以及测试前向传播过程中，batch_x对应的标签处理方式相同)；
        而对应在oracles_yi为真实类标向量

        注意：每次batch训练集数据的大小均为[n_way*num_shots]*batch_size，为何对于batches_xi要存放为大小为n_way*num_shots的list，每个数据
        的维度为batch_size呢？这里是为了与batch_x的维度进行统一，方便提取图像特征的网络前向传播
        '''
        batch_x = np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32')
        labels_x = np.zeros((batch_size, n_way), dtype='float32')
        labels_x_global = np.zeros(batch_size, dtype='int64')
        target_distances = np.zeros((batch_size, n_way * num_shots), dtype='float32')
        hidden_labels = np.zeros((batch_size, n_way * num_shots + 1), dtype='float32')
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way*num_shots): # n_way*num_shots表示每次迭代一共有多少个样本
            batches_xi.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
        # Iterate over tasks for the same batch

        for batch_counter in range(batch_size):
            positive_class = random.randint(0, n_way - 1) # 在n_ways种类中选择一类，取其中一个样本作为测试样本

            # Sample random classes for this TASK
            classes_ = list(self.data.keys())
            # 从classes_中提取n_wayes个不同的数据，作为该次迭代所选择的n_ways个类别
            sampled_classes = random.sample(classes_, n_way)
            indexes_perm = np.random.permutation(n_way * num_shots) # 对np.arange(n_way * num_shots)随机重排

            counter = 0
            # 对于每一个被选择的类
            for class_counter, class_ in enumerate(sampled_classes):
                # samples表示该类中被用来训练的数据
                if class_counter == positive_class: # 如果是测试样本所在的类别
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_], num_shots+1) # 从self.data[class_]提取num_shots+1个不同的数据
                    # Test sample is loaded
                    batch_x[batch_counter, :, :, :] = samples[0]
                    labels_x[batch_counter, class_counter] = 1
                    labels_x_global[batch_counter] = self.class_encoder[class_]
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_], num_shots)

                for s_i in range(0, len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :, :, :] = samples[s_i]
                    # 从每类中选择unlabeled_extra个额外数据
                    if s_i < unlabeled_extra:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 0
                        hidden_labels[batch_counter, indexes_perm[counter] + 1] = 1
                    else:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(positive_class)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1) # 得到的大小为batch_size

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi,
                      torch.from_numpy(hidden_labels)]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr

    def cast_cuda(self, input):
        if type(input) == type([]): # 如果输入的是list
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def cast_variable(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_variable(input[i])
        else:
            return Variable(input)

        return input
