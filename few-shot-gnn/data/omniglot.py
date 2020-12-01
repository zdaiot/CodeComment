from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import numpy as np
from PIL import Image as pil_image
import pickle
import random
from . import parser


class Omniglot(data.Dataset):
    def __init__(self, root, dataset='omniglot'):
        self.root = root
        self.seed = 10
        self.dataset = dataset
        if not self._check_exists_(): # 若不存在compacted_datasets以及对应文件
            self._init_folders_()
            if self.check_decompress(): # 若omniglot/test为空文件夹
                self._decompress_()
            self._preprocess_()

    # 创建需要的文件夹
    def _init_folders_(self):
        decompress = False
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root, 'omniglot')):
            os.makedirs(os.path.join(self.root, 'omniglot'))
            decompress = True
        if not os.path.exists(os.path.join(self.root, 'omniglot', 'train')):
            os.makedirs(os.path.join(self.root, 'omniglot', 'train'))
            decompress = True
        if not os.path.exists(os.path.join(self.root, 'omniglot', 'test')):
            os.makedirs(os.path.join(self.root, 'omniglot', 'test'))
            decompress = True
        if not os.path.exists(os.path.join(self.root, 'compacted_datasets')):
            os.makedirs(os.path.join(self.root, 'compacted_datasets'))
            decompress = True
        return decompress

    def check_decompress(self):
        return os.listdir('%s/omniglot/test' % self.root) == []

    # 解压images_background.zip和images_evaluation.zip到omniglot/train和omniglot/test
    def _decompress_(self):
        print("\nDecompressing Images...")
        comp_files = ['%s/compressed/omniglot/images_background.zip' % self.root,
                      '%s/compressed/omniglot/images_evaluation.zip' % self.root]
        if os.path.isfile(comp_files[0]) and os.path.isfile(comp_files[1]):
            os.system(('unzip %s -d ' % comp_files[0]) +
                      os.path.join(self.root, 'omniglot', 'train')) # os.system执行系统指令
            os.system(('unzip %s -d ' % comp_files[1]) +
                      os.path.join(self.root, 'omniglot', 'test'))
        else:
            raise Exception('Missing %s or %s' % (comp_files[0], comp_files[1]))
        print("Decompressed")

    def _check_exists_(self):
        return os.path.exists(os.path.join(self.root, 'compacted_datasets', 'omniglot_train.pickle')) and \
               os.path.exists(os.path.join(self.root, 'compacted_datasets', 'omniglot_test.pickle'))

    def _preprocess_(self):
        print('\nPreprocessing Omniglot images...')
        (class_names_train, images_path_train) = parser.get_image_paths(os.path.join(self.root, 'omniglot', 'train'))
        (class_names_test, images_path_test) = parser.get_image_paths(os.path.join(self.root, 'omniglot', 'test'))

        keys_all = sorted(list(set(class_names_train + class_names_test))) # sorted对类别按照字母先后顺序排序
        label_encoder = {}
        label_decoder = {}
        for i in range(len(keys_all)):
            label_encoder[keys_all[i]] = i # 对类别进行编码及解码，从0开始
            label_decoder[i] = keys_all[i]

        all_set = {}
        for class_, path in zip(class_names_train + class_names_test, images_path_train + images_path_test):
            img = np.array(pil_image.open(path), dtype='float32')
            if label_encoder[class_] not in all_set:
                all_set[label_encoder[class_]] = [] # 在all_set字典中创建键值label_encoder[class_]，并置空
            all_set[label_encoder[class_]].append(img) # 将img添加到键值label_encoder[class_]中

        # Now we save the 1200 training - 423 testing partition
        keys = sorted(list(all_set.keys())) # 不同于keys_all，这里的均是数字
        random.seed(self.seed)
        random.shuffle(keys)
        # train_set和test_set均为dict，分别存放1200和423个键值，每一个键值表示一类，每类有20个数据
        train_set = {}
        test_set = {}
        for i in range(1200):
            train_set[keys[i]] = all_set[keys[i]]
        for i in range(1200, len(keys)):
            test_set[keys[i]] = all_set[keys[i]]

        self.sanity_check(all_set) # 判断是否所有类都有20个样本
        # 存放训练集与测试集（dict）
        with open(os.path.join(self.root, 'compacted_datasets', 'omniglot_train.pickle'), 'wb') as handle:
            pickle.dump(train_set, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted_datasets', 'omniglot_test.pickle'), 'wb') as handle:
            pickle.dump(test_set, handle, protocol=2)
        # 存放类别与编码之间的映射关系（dict）
        with open(os.path.join(self.root, 'compacted_datasets', 'omniglot_label_encoder.pickle'), 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted_datasets', 'omniglot_label_decoder.pickle'), 'wb') as handle:
            pickle.dump(label_decoder, handle, protocol=2)
        
        print('Images preprocessed')

    def sanity_check(self, all_set):
        all_good = True
        for class_ in all_set:
            if len(all_set[class_]) != 20:
                all_good = False
        if all_good:
            print("All classes have 20 samples")

    # 加载train或者test数据集，并按照size的大小resize
    def load_dataset(self, train, size):
        print("Loading dataset")
        if train:
            with open(os.path.join(self.root, 'compacted_datasets', 'omniglot_train.pickle'), 'rb') as handle:
                data = pickle.load(handle)
        else:
            with open(os.path.join(self.root, 'compacted_datasets', 'omniglot_test.pickle'), 'rb') as handle:
                data = pickle.load(handle)
        print("Num classes before rotations: "+str(len(data)))

        data_rot = {}
        # resize images and normalize
        for class_ in data:
            for rot in range(4): # 每一类所有图片均旋转0,90,180,270度，组成新的4类
                data_rot[class_ * 4 + rot] = []
                for i in range(len(data[class_])):
                    image2resize = pil_image.fromarray(np.uint8(data[class_][i]*255))
                    image_resized = image2resize.resize((size[1], size[0]))
                    image_resized = np.array(image_resized, dtype='float32')/127.5 - 1 # 将0~255等比例归一化到-1~1
                    image = self.rotate_image(image_resized, rot)
                    image = np.expand_dims(image, axis=0) # 若image为[20,20]，则执行后变成[1,20,20]
                    data_rot[class_ * 4 + rot].append(image)

        print("Dataset Loaded")
        print("Num classes after rotations: "+str(len(data_rot)))
        self.sanity_check(data_rot)
        return data_rot

    def rotate_image(self, image, times):
        rotated_image = np.zeros(image.shape)
        for channel in range(image.shape[0]):
            rotated_image[:, :] = np.rot90(image[:, :], k=times)
        return rotated_image