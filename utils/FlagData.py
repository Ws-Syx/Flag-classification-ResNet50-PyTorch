# 导入相关模块
from random import shuffle

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io, transform
from PIL import Image
import numpy as np
import random
import os
import torch


def save_dict(filename, dictionary):
    # print('dictionary = {}'.format(dictionary))
    labels = np.array(list(dictionary.keys()))
    # print('label = {}'.format(labels))
    np.save(filename, labels)


def load_dict(filename):
    labels = np.load(filename)
    return {name: index for index, name in zip(range(len(labels)), labels)}


def data_split(root_path, rate):
    train_dataset = []
    test_dataset = []

    listdir = os.listdir(root_path)
    for class_name in listdir:
        class_path = root_path + '/' + class_name
        # print('class_path = {}'.format(class_path))
        if os.path.isdir(class_path):
            # 获取当前文件夹下所有的图片文件的绝对地址
            image_list = []
            for file_name in os.listdir(class_path):
                file_name = root_path + '/' + class_name + '/' + file_name
                if os.path.isfile(file_name):
                    if file_name[-4:] == '.jpg' or file_name[-4:] == '.JPG':
                        image_list.append(file_name)
                    elif file_name[-5:] == '.jpeg' or file_name[-5:] == '.JPEG':
                        image_list.append(file_name)
                    elif file_name[-4:] == '.png' or file_name[-4:] == '.PNG':
                        image_list.append(file_name)

            # 对当前目录下所有的图片文件切割成两部分
            length = len(image_list)
            index = np.arange(0, length)
            random.shuffle(index)

            train_index = index[:int(length * rate)]
            test_index = index[int(length * rate):]

            for i in train_index:
                train_dataset.append([image_list[i], class_name])
            for i in test_index:
                test_dataset.append([image_list[i], class_name])

    if len(train_dataset) > 0:
        np.savetxt(root_path + '/train_dataset.txt', train_dataset, encoding='utf-8', fmt='%s, %s')
    if len(test_dataset):
        np.savetxt(root_path + '/test_dataset.txt', test_dataset, encoding='utf-8', fmt='%s, %s')

    return root_path + '/train_dataset.txt', root_path + '/test_dataset.txt'


class FlagData(Dataset):  # 继承Dataset
    def __init__(self, path, transform=None, use_dict=None):
        # 变换
        self.transform = transform

        # 数据集标记文件存在
        if os.path.isfile(path):
            # 读取图片路经
            self.values = np.loadtxt(path, delimiter=',', usecols=[0], encoding='utf-8', dtype=np.str)
            # 读取分类标签
            self.labels = np.loadtxt(path, delimiter=',', usecols=[1], encoding='utf-8', dtype=np.str)
            # 打乱顺序
            random_index = np.arange(0, len(self.labels))
            random.shuffle(random_index)
            self.values = self.values[random_index]
            self.labels = self.labels[random_index]

            if use_dict is None:
                class_names = list(set(self.labels))
                # print('1. class_names = {}'.format(class_names))
                class_names = sorted(class_names)
                # print('2. class_names = {}'.format(class_names))
                self.label_to_index = {name: index for index, name in zip(range(len(class_names)), class_names)}
                save_dict('./model/dict.npy', self.label_to_index)
            elif use_dict is not None:
                self.label_to_index = load_dict(use_dict)

        else:
            print('划分文件不存在')

    def __len__(self):  # 返回整个数据集的大小
        return len(self.labels)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # 获取图片
        image_path = self.values[index]
        image = Image.open(image_path)
        # 获取label
        label = self.labels[index]

        # 对样本进行变换
        if self.transform:
            image = self.transform(image)
        # 对label进行变换
        # case 1
        # target = torch.zeros(len(self.label_to_index))
        # target[self.label_to_index[label]] = 1
        # print('sum of target = {}'.format(target.numpy().sum()))
        # target = torch.LongTensor(target.numpy())
        # case 2
        target = self.label_to_index[label]
        return image, target


# if __name__ == '__main__':
#     print("==> Prepairing train_dataset ...")
#
#     trans = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#
#     # 切割训练集、测试集
#     train_dataset_path, test_dataset_path = data_split('./../train_dataset')
#
#     # dataset
#     train_dataset = FlagData(train_dataset_path, trans)
#     test_dataset = FlagData(test_dataset_path, trans)
#
#     # dataloader
#     train_loader = torch.utils.train_dataset.DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=2, drop_last=False)
#     test_loader = torch.utils.train_dataset.DataLoader(test_dataset, batch_size=30, shuffle=True, num_workers=2, drop_last=False)
#
#     print('load successfully')
#     print(train_dataset.label_to_index)
#
#     for i, (x, y) in enumerate(train_loader):
#         qwq = 0
#         # print('x = {}'.format(x))
#         # print('y = {}'.format(y))
#
#     # dict = load_dict('dict.npy')
#     # print(dict)
