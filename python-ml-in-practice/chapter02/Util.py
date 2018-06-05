# encoding:utf-8
import numpy as np


class DataUtil:
    # 定义一个方法能使其从文件中读取数据
    # 该方法接收五个参数：
    # 数据集的名字、数据集路径、训练样本数、类别所在列、是否打乱数据
    def get_dataset(name, path, train_num=None, tar_ind=None, shuffle=True):
        x = []
        # 将编码设为utf8，以便读入中文等特殊字符
        with open(path, 'r', encoding='utf8') as file:
            # 如果是气球数据集，直接依据逗号分隔即可
            if 'balloon' in name or 'mushroom' in name:
                for sample in file:
                    x.append(sample.strip().split(','))
        # 默认打乱数据
        if shuffle:
            np.random.shuffle(x)

        # 默认类别在最后一列
        tar_ind = -1 if tar_ind is None else tar_ind
        y = np.array([xx.pop(tar_ind) for xx in x])
        x = np.array(x)

        # 默认全部是训练样本
        if train_num is None:
            return x, y
        # 若是传入了训练样本数，则将数据集切分为训练集和测试集
        return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])


