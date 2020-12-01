import os
import fnmatch


def get_image_paths(source, extension='png'):
    images_path, class_names = [], []
    '''
    os.work用法：root先取source路径，dirnames和filenames分别为当前root下的文件夹和文件；下次for循环，root取
    source/子文件夹，此时，dirnames和filenames分别为当前root下的文件夹和文件；直到root遍历完所有的文件夹停止
    '''
    for root, dirnames, filenames in os.walk(source):
        '''
        当root等于'../datasets/omniglot/train/images_background/Japanese_(katakana)/character21'，dirnames
        为[]时，filename有值，值为['0616_10.png', '0616_04.png', ...]
        '''
        filenames = [filename for filename in filenames if '._' not in filename]
        for filename in fnmatch.filter(filenames, '*.'+extension): # 过滤只以.png结尾的文件
            images_path.append(os.path.join(root, filename)) # 当前文件的路径
            class_name = root.split('/')
            class_name = class_name[len(class_name)-2:] # 取list中最后两个元素，例如['Japanese_(katakana)', 'character21']
            class_name = '/'.join(class_name)
            class_names.append(class_name) # 当前文件的类标
    return class_names, images_path

if __name__ == "__main__":
    (class_names_train, images_path_train) = get_image_paths(os.path.join('../datasets', 'omniglot', 'train'))
    (class_names_test, images_path_test) = get_image_paths(os.path.join('../datasets', 'omniglot', 'test'))
    keys_all = sorted(list(set(class_names_train + class_names_test)))
    print(keys_all)