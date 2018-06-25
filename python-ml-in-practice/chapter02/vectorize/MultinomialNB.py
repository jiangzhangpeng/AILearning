# encoding:utf-8
import numpy as np

from Basic import NaiveBayes


# 导入基本架构Basic
class MultinomialNB(NaiveBayes):
    # 定义预处理数据的方法
    def feed_data(self, x, y, sample_weight=None):
        # 分情况将输入的x向量进行转置
        if isinstance(x, list):
            features = map(list, zip(*x))
        else:
            features = x.T
        # 利用python的高级数据结构--集合，获取各个维度的特征和类别种类
        # 为了利用bincount方法来优化算法，将所有特征从0开始数值化
        # 注意：需要将数值化过程中的转换关系记录为字典，否则无法对新数据进行判断
        features = [set(feat) for feat in features]
        feat_dics = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        label_dics = {_l: i for i, _l in enumerate(set(y))}
        # 利用转换字典更新训练集
        x = np.array([[feat_dics[i][_l] for i, _l in enumerate(sample)] for sample in x])
        y = np.array([label_dics[yy] for yy in y])
        # 利用bincount方法，获取各类别的数据的个数
        cat_counter = np.bincount(y)
        # 记录各维度特征的取值个数
        n_possibilities = [len(feats) for feats in features]
        # 获取各类别数据下标
        labels = [y == value for value in range(len(cat_counter))]
        # 利用下标获取记录按类别分开后的输入数据的数组
        labelled_x = [x[ci].T for ci in labels]
        # 更新模型的各个属性
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        (self._cat_counter, self._feat_dics, self._n_possibilities) = (cat_counter, feat_dics, n_possibilities)
        self.label_dic = {i: _l for _l, i in label_dics.items()}
        # 调用处理样本权重的函数，以便更新记录条件概率的数组
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        # 利用numpy的bincount方法获取带权重条件的概率的极大似然估计
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([np.bincount(xx[dim], minlength=_p) for xx in self._labelled_x])
            else:
                self._con_counter.append(
                    [np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=_p) for
                     label, xx in self._label_zip])

    # 定义核心训练函数
    def _fit(self, lb):
        n_dim = len(self._n_possibilities)
        n_category = len(self._cat_counter)
        p_catefory = self.get_prior_probability(lb)
        # data即为粗出加了平滑项后的条件概率数组
        data = [None] * n_dim
        for dim, n_possibilities in enumerate(self._n_possibilities):
            # 对每一个feature（dim），按照feature的取值个数及分类的取值个数，分别计算条件概率
            # 比如feature取值a，b，分类为x,y，分别计算[[f(a|x),f(b|x)],[f(a|y),f(b|y)]]
            data[dim] = [[(self._con_counter[dim][c][p] + lb) / (self._cat_counter[c] + lb * n_possibilities) for p in
                          range(n_possibilities)] for c in range(n_category)]
            self._data = [np.array(dim_info) for dim_info in data]
            # 利用data生成决策函数
        '''
        def func(input_x, tar_category):
            rs = 1
            # 遍历各个维度，利用data和条件独立性假设计算联合条件概率
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category][xx]
            # 利用先验概率和联合条件概率计算后验概率
            return rs * p_catefory[tar_category]
            '''

        # 向量化决策函数
        def func(input_x, tar_category):
            # 将输入转化为二维数组
            input_x = np.atleast_2d(input_x).T
            # 使用向量存储结果，全部初始化为1
            rs = np.ones(input_x.shape[1])
            for d, xx in enumerate(input_x):
                # 向量操作
                rs *= self._data[d][tar_category][xx]
            return rs * p_catefory[tar_category]

        # 返回决策函数
        return func
    '''
    # 定义数值化数据的函数
    def _transfer_x(self, x):
        # 遍历每个元素，利用转换字典进行数值化
        for j, char in enumerate(x):
            x[j] = self._feat_dics[j][char]
        return x
    '''

    # 定义数值化数据的函数  向量化改造
    def _transfer_x(self, x):
        # 遍历每个元素，利用转换字典进行数值化
        for xx in x:
            for j, char in enumerate(xx):
                xx[j] = self._feat_dics[j][char]
        return x

if __name__ == '__main__':
    # 导入time计时，导入DataUtil获取数据
    import time
    from Util import DataUtil

    # 遍历1.0,1.5两个版本的气球数据
    # for dataset in ('balloon1.0', 'balloon1.5'):
    for dataset in ['mushroom']:
        # 读取数据
        # _x, _y = DataUtil.get_dataset(dataset, '{}.txt'.format(dataset))
        _x, _y = DataUtil.get_dataset(dataset, '{}.txt'.format(dataset), tar_ind=0)
        # 实例化模型并进行训练，同时记录过程发生的时间
        learning_time = time.time()
        nb = MultinomialNB()
        nb.fit(_x, _y)
        learning_time = time.time() - learning_time
        # 评估训练模型的表现，同时记录评估过程发生的时间
        estimate_time = time.time()
        nb.evaluate(_x, _y)
        estimate_time = time.time() - estimate_time
        # 将记录下来的耗时输出
        print(
            'Model building:{:12.6} s\n'
            'Estimation    :{:12.6} s\n'
            'Total         :{:12.6} s'.format(learning_time, estimate_time, learning_time + estimate_time)
        )

    # 导入matplotlib库进行可视化
    import matplotlib.pyplot as plt
    # 进行一些设置，使得matplotlib可以显示中文
    from pylab import mpl

    # 将字体设为仿宋
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
    # 利用Multinomial搭建过程中记录的变量获取条件概率
    data = nb['data']
    # 定义颜色字典，将类别e（能吃）设为天蓝色，类别p（有毒）设为橙色
    colors = {'e': 'lightSkyBlue', 'p': 'orange'}
    # 利用转换字典定义其“反字典”，后面可视化会用上
    _rev_feat_dics = [{_val: _key for _key, _val in _feat_dics.items()} for _feat_dics in nb._feat_dics]
    # 遍历各维度进行可视化
    # 利用MultinomialNB搭建过程中记录的变量，获取画图需要的信息
    for _j in range(nb['x'].shape[1]):
        sj = nb['n_possibilities'][_j]
        tmp_x = np.arange(1, sj + 1)
        # 利用matplotlib对latex的支持来写标题，两个$之间即是latex语句
        # 标题为第几个feature(_j+1)，共有几种取值(sj)
        title = '$j = {};s_j = {}$'.format(_j + 1, sj)
        plt.figure()
        plt.title(title)
        # 根据条件概率的大小画出柱状图
        # 针对每个feature的每个category画图
        for _c in range(len(nb.label_dic)):
            plt.bar(tmp_x - 0.35 * _c, data[_j][_c, :], width=0.35, facecolor=colors[nb.label_dic[_c]],
                    edgecolor='white', label='class:{}'.format(nb.label_dic[_c]))
            # 利用上文的反字典将横坐标转化成特征的各个取值
            plt.xticks([i for i in range(sj + 2)], [''] + [_rev_feat_dics[i] for i in range(sj)] + [''])
            plt.ylim(0, 1.0)
            plt.legend()
            # 保存画好的图像
            plt.savefig('pics/d{}'.format(_j + 1))
