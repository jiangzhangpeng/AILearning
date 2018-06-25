# encoding:utf-8
import copy
import time

import numpy as np
from Node import CvDNode
from Util import DataUtil


# 实现一个足够抽象的tree结构的基类以适应我们的node结构基类
class CVDBase:
    """
    初始化结构
     self.nodes 记录所有的node的列表
     self.roots 主要用于cart的剪枝的属性
     self.max_depth 记录决策树最大深度的属性
     self.root, self.feature_sets 根节点和记录可选特征维度的列表
     self.label_dic 类别转换字典
     self.prune_alpha， self.layers 主要用于ID3和C4.5剪枝的两个属性 惩罚因子  记录每一层的node

    """

    def __init__(self, max_depth=None, node=None):
        self.nodes, self.layers, self.roots = [], [], []
        self.max_depth = max_depth
        self.root = node
        self.feature_sets = []
        self.label_dic = {}
        self.prune_alpha = 1
        self.whether_continuous = None

    def __str__(self):
        return 'CvDTree ({})'.format(self.root.height)

    __repr__ = __str__

    def feed_data(self, x, continuous_rate=0.2):
        # 利用set获取各个维度特征的所有可能取值
        self.feature_sets = [set(dimension) for dimension in x.T]
        data_len, data_dim = x.shape
        # 判断是否连续
        self.whether_continuous = np.array([len(feat) >= continuous_rate * data_len for feat in self.feature_sets])
        self.root.feats = [i for i in range(x.shape[1])]
        self.root.feed_tree[self]

    # 参数alpha和剪枝有关，可以先不说明
    # cv_rate用于控制验证集大小，train_only控制程序是否进行数据集的切分
    def fit(self, x, y, alpha=None, sample_weight=None, eps=1e-8, cv_rate=0.2, train_only=False):
        # 数值化类别向量
        _dic = {c: i for i, c in enumerate(set(y))}
        y = np.array([_dic[yy] for yy in y])
        self.label_dic = {value: key for key, value in _dic.items()}
        x = np.array(x)
        # 根据特征个数定出alpha
        self.prune_alpha = alpha if alpha is not None else x.shape[1] / 2
        # 如果需要划分数据集的话
        if not train_only and self.root.is_cart:
            # 根据cv_rate将数据集随机分为训练集和交叉验证集
            # 实现的核心思想是利用下表来进行各种切分
            _train_num = int(len(x) * (1 - cv_rate))
            _indices = np.random.permutation(np.arange(len(x)))
            _train_indices = _indices[:_train_num]
            _test_indices = _indices[_train_num:]
            if sample_weight is not None:
                # 注意对切分后的样本权重做归一化处理
                _train_weights = sample_weight[_train_indices]
                _test_weights = sample_weight[_test_indices]
                _train_weights /= np.sum(_train_weights)
                _test_weights /= np.sum(_test_weights)
            else:
                _test_weights = _train_weights = None
            x_train, y_train = x[_train_indices], y[_train_indices]
            x_cv, y_cv = x[_test_indices], y[_test_indices]
        else:
            x_train, y_train, _train_weights = x, y, sample_weight
            cv_x = cv_y = _test_weights = None
        self.feed_data(x_train)
        # 调用跟节点的生成算法
        self.root.fit(x_train, y_train, _train_weights, eps)
        # 调用对Node剪枝算法的封装
        self.prune(x_cv, y_cv, _test_weights)

    def reduce_nodes(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            if self.nodes[i].pruned:
                self.nodes.pop(i)

    def _update_layers(self):
        # 根据整棵决策树的深度，在self.layers里面放相应数量的列表
        self.layers = [[] for _ in range(self.root.height)]
        self.root.update_layers()

    def _prune(self):
        self._update_layers()
        _tmp_nodes = []
        # 更新完决策树每一层的node之后，从后往前向_tmp_nodes中加node
        for _node_lst in self.layers[::-1]:
            for _node in _node_lst[::-1]:
                if _node.category is None:
                    _tmp_nodes.append(_node)
        _old = np.array([node.cost() + self.prune_alpha * len(node.leafs) for node in _tmp_nodes])
        _new = np.array([node.cost(pruned=True) + self.prune_alpha for node in _tmp_nodes])
        # 使用_mask变量存储_old和_new对应位置的关系
        _mask = _old >= _new
        while True:
            # 若只剩下根节点就退出循环体
            if self.root.height == 1:
                return
            p = np.argmax(_mask)
            # 如果_new中有比_old中对应的损失小的损失，则进行局部剪枝
            if _mask[p]:
                _tmp_nodes[p].prune()
                # 根据剪枝影响了的Node，更新_old，_mask对应位置的值
                for i, node in enumerate(_tmp_nodes):
                    if node.affected:
                        _old[i] = node.cost() + self.prune_alpha * len(node.leafs)
                        _mask[i] = _old[i] >= _new[i]
                        node.affected = False
                # 根据被剪掉的Node，将各个变量对应的位置除去（除以从后往前遍历）
                for i in range(len(_tmp_nodes) - 1, -1, -1):
                    if _tmp_nodes[i].pruned:
                        _tmp_nodes.pop(i)
                        _old = np.delete(_old, i)
                        _new = np.delete(_new, i)
                        _mask = np.delete(_mask, i)
            else:
                break
        self.reduce_nodes()

    def _cart_prune(self):
        # 暂时将所有节点记录所属tree的属性置位None
        self.root.cut_tree()
        _tmp_nodes = [node for node in self.nodes if node.catefory is None]
        _thresholds = np.array([node.get_threshold() for node in _tmp_nodes])
        while True:
            # 利用deepcopy对于当前节点进行深拷贝，存入self.roots列表
            # 如果前面没有把记录tree的属性置为none，那么这个也会对整个tree做深度拷贝。
            # 可以想象，这会引发严重的内存问题，速度也会被拖的非常慢
            root_copy = copy.deepcopy(self.root)
            self.roots.append(root_copy)
            if self.root.height == 1:
                break
            p = np.argmin(_thresholds)
            _tmp_nodes[p].prune()
            for i, node in enumerate(_tmp_nodes):
                # 更新影响node的阈值
                if node.affected:
                    _thresholds[i] = node.get_threshold()
                    node.affected = False
            for i in range(len(_tmp_nodes) - 1, -1, -1):
                # 去掉个列表相对应位置的原色
                if _tmp_nodes[i].pruned:
                    _tmp_nodes.pop(i)
                    _thresholds = np.delete(_thresholds, i)
        return self.reduce_nodes()

    def cut_tree(self):
        self.tree = None
        for child in self.children.values():
            if child is not None:
                child.cut_tree()

    # 定义计算加权正确率的函数
    def acc(self, y, y_pred, weights):
        if weights is not None:
            return np.sum((np.array(y) == np.array(y_pred)) * weights) / len(y)
        return np.sum(np.array(y) == np.array(y_pred)) / len(y)

    def prune(self, x_cv, y_cv, weights):
        if self.root.is_cart:
            # 如果该node使用cart剪枝，那么只有确实传入交叉验证集的情况下才能调用相关函数，否则没有意义
            if x_cv is not None and y_cv is not None:
                self._cart_prune()
                _arg = np.argmax([CVDBase.acc(y_cv, tree.predict(x_cv), weights) for tree in self.roots])
                _tar_root = self.roots[_arg]
                # 由于node的feed_tree方法会通过递归更新nodes属性，所以要先充值
                self.nodes = []
                _tar_root.feed_tree(self)
                self.root = _tar_root
        else:
            self._prune()

    def predict_one(self, x):
        if self.category is not None:
            return self.category
        if self.is_continuous:
            if x[self.feature_dim] < self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        if self.is_cart:
            if x[self.feature_dim] == self.tar:
                return self.left_child.predict_one(x)
            return self.right_child.predict_one(x)
        else:
            try:
                return self.children[x[self.feature_dim]].predict_one(x)
            except KeyError:
                return self.get_category()

    def predict(self, x):
        return np.array([self.predict_one(xx) for xx in x])


# 在CVDNode的基础上，定义ID3，C4.5和CART算法对应的Node结构
class ID3Tree(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = 'ent'


class C45Tree(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = 'ratio'


class CartTree(CvDNode):
    def __init__(self, *args, **kwargs):
        CvDNode.__init__(self, *args, **kwargs)
        self.criterion = 'gini'
        self.is_cart = True


def main(visualize=True):
    # x, y = DataUtil.get_dataset("balloon1.0(en)", "../_Data/balloon1.0(en).txt")
    x, y = DataUtil.get_dataset('test', 'test.txt')
    fit_time = time.time()
    #tree = CartTree(whether_continuous=[False] * 4)
    tree = CartTree()
    #tree.fit(x, y, train_only=True)
    tree.fit(x, y,sample_weight = None)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x, y)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize()

    train_num = 6000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "mushroom", "../_Data/mushroom.txt", tar_idx=0, n_train=train_num)
    fit_time = time.time()
    tree = C45Tree()
    tree.fit(x_train, y_train)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x_train, y_train)
    tree.evaluate(x_test, y_test)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize()

    x, y = DataUtil.gen_xor(one_hot=False)
    fit_time = time.time()
    tree = CartTree()
    tree.fit(x, y, train_only=True)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x, y, n_cores=1)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize2d(x, y, dense=1000)
        tree.visualize()

    wc = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for _cl in continuous_lst:
        wc[_cl] = True

    train_num = 2000
    (x_train, y_train), (x_test, y_test), *_ = DataUtil.get_dataset(
        "bank1.0", "../_Data/bank1.0.txt", n_train=train_num, quantize=True)
    fit_time = time.time()
    tree = CartTree()
    tree.fit(x_train, y_train)
    fit_time = time.time() - fit_time
    if visualize:
        tree.view()
    estimate_time = time.time()
    tree.evaluate(x_test, y_test)
    estimate_time = time.time() - estimate_time
    print(
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            fit_time, estimate_time,
            fit_time + estimate_time
        )
    )
    if visualize:
        tree.visualize()

    tree.show_timing_log()


if __name__ == '__main__':
    main(False)
    print('done!')
