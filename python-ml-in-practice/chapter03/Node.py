# encoding:utf-8
import numpy as np
from Cluster import Cluster


# 定义一个足够抽象的基类以囊括所有我们关心的算法
class CvDNode:
    """
    初始化结构
    self._x , self._y 记录数据集的变量
    self.base, self.chaos 记录对数的底和当前的不确定性
    self.criterion , self.category 记录该节点计算信息增益的方法和所属的类别
    self.left_child , self.right_child 针对连续性特征和CART，记录该节点的左右子节点
    self._children, self.leafs 记录该node所有子节点和下属的叶子节点
    self.sample_weight 记录样本权重
    self.wc 记录各特征是否连续 whether continuous
    self.tree 记录该node所属的tree
    self.feature_dim, self.tar, self.feats 记录该node划分标准的相关信息，具体如下：
        self.feature_dim 记录作为划分标准的特征所对应的维度j*
        self.tar 针对连续型和CART，记录二分标准
        elf.feats 记录该node进行选择的，作为划分标准的特征的维度
        self.parent, self.is_root 记录该node的父节点以及该node是否为根节点
        self._depth, self.prev_feat 记录该node的深度以及其父节点的划分标准
        self.is_cart 是否使用了cart算法
        self.is_continuous 记录该node的划分标准对应的特征是否为连续
        self.pruned  记录该node是否已被剪掉，后面实现局部剪枝算法会用到
    """

    def __init__(self, tree=None, base=2, chaos=None, depth=0, parent=None, is_root=True, prev_feat='Root'):
        self._x = self._y = None
        self.base, self.chaos = base, chaos
        self.criterion = self.category = None
        self.left_child = self.right_child = None
        self._children, self.leafs = {}, {}
        self.sample_weight = None
        self.wc = None
        self.tree = tree
        # 如果传入了tree，进行相应的初始化
        if tree is not None:
            # 由于数据的预处理是由Tree完成的
            # 所以各个维度的特征是否连续性随机变量也是由Tree记录的
            self.wc = tree.whether_continuous
            # 这里的nodes变量是tree中记录所有node的列表
            tree.nodes.append(self)
        self.feature_dim, self.tar, self.feats = None, None, []
        self.parent, self.is_root = parent, is_root
        self._depth, self.prev_feat = depth, prev_feat
        self.is_cart = self.is_continuous = self.pruned = False

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, '_' + item)

    # 重载__lt__方法，使得node之间可以比较谁更小，进而方便调试和可视化
    def __lt__(self, other):
        return self.prev_feat < other.prev_feat

    # 重载__str__和__repr__方法，同样是为方便调试和可视化
    def __str__(self):
        if self.category is None:
            return 'CvDNode ({}) ({} --> {})'.format(self._depth, self.prev_feat, self.feature_dim)
        return 'CvDNode ({}) ({} --> class:{})'.format(self._depth, self.prev_feat, self.tree.label_dic[self.category])

    __repr__ = __str__

    # 定义children属性，主要是区分连续+cart和其余情况，有了该属性后，想要获得所有子节点时就不用分情况讨论了
    @property
    def children(self):
        return {'left': self.left_child, 'right': self.right_child} if (
                self.is_cart or self.is_continuous) else self._children

    # 低估定义height属性，叶子节点高度定义为1，其余节点高度定义为最高的子节点的高度+1
    @property
    def height(self):
        if self.category is not None:
            return 1
        return 1 + max([_child.height if _child is not None else 0 for _child in self._children.values()])

    # 定义info_dic(信息字典)属性，他记录了该Node的主要信息
    # 在更新各个node的叶子节点时，被记录进各个self.leafs属性的就是该字典
    @property
    def info_dic(self):
        return {'chaos': self.chaos, 'y': self._y}

    # 定义第一种停止准则：当特征维度为0，或当前node的数据不准确性小于阈值时停止
    # 同时，如果用户指定了决策树的最大深度，那么当该node的深度太深时也停止
    # 若满足了停止条件，该函数会返回True没否则返回false
    def stop1(self, eps):
        if (self._x.shape[1] == 0 or (self.chaos is not None and self.chaos < eps)
                or (self.tree is not None and self.tree.max_depth is not None and self._depth >= self.tree.max_depth)):
            # 调用处理停止情况的方法
            self._handle_terminate()
            return True
        return False

    # 定义第二种停止准则，当最大信息增益仍然小于阈值是停止
    def stop2(self, max_gain, pes):
        if max_gain <= eps:
            self._handle_terminate()
            return True
        return False

    # 利用bincount方法定义根据数据生成该node所属类别的方法
    def get_category(self):
        return np.argmax(np.bincount(self._y))

    # 定义处理停止情况的方法，核心思想就是把该node转化成一个叶子节点
    def _handle_terminate(self):
        # 首先要生成该node所属类别
        self.category = self.get_category()
        # 然后一路回溯，更新父节点、父节点的父节点，等等，记录叶子几点的属性leafs
        _parent = self.parent
        while _parent is not None:
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent

    def prune(self):
        # 调用相应的方法计算该node所属类别
        self.category = self.get_category()
        # 计算该node转化为叶子节点而被剪去的，下属的叶子节点
        _pop_list = [key for key in self.leafs]
        # 然后一路回溯，更新各个parent的属性leafs（使用id作为key以避免重复）
        _parent = self.parent
        while _parent is not None:
            # 删去由于局部剪枝而被减掉的叶子节点
            for _k in _pop_list:
                _parent.leafs.pop(_k)
            _parent.leafs[id(self)] = self.info_dic
            _parent = _parent.parent
        # 调用mark_pruned方法将自己所有的子节点，子节点的子节点。。。。pruned属性置位True，因为他们都被减掉了
        self.mark_pruned()
        # 重置各个属性
        self.feature_dim = None
        self.left_child = self.right_child = None
        self._children = {}
        self.leafs = {}

    def mark_pruned(self):
        self.pruned = True
        # 遍历各个子节点
        for _child in self.children.values():
            # 如果子节点不是none的话，递归调用该方法
            # 连续性和cart的子节点可能为空，因为他们的子节点为left和right，没有存在childrend中
            if _child is not None:
                _child.mark_pruned()

    def fit(self, x, y, sample_weight, eps=1e-8):
        self._x, self._y = np.atleast_2d(x), np.array(y)
        self.sample_weight = sample_weight
        # 若满足第一种停止准则，则退出函数体
        if self.stop1(eps):
            return
        # 用该node的数据实例化cluster类以计算各种信息量
        _cluster = Cluster(self._x, self._y, sample_weight, self.base)
        # 对于根节点，需要额外计算其数据的不确定性
        if self.is_root:
            if self.criterion == 'gini':
                self.chaos = _cluster.gini()
            else:
                self.chaos = _cluster.ent()
        _max_gain, _chaos_list = 0, []
        _max_feature = _max_tar = None
        # 遍历还能选择的特征
        for feat in self.feats:
            # 如果是连续性或者cart算法，需要额外计算二分标准的取值集合
            if self.wc[feat]:
                _samples = np.sort(self._x.T[feat])
                _set = (_samples[:-1] + _samples[1:]) * 0.5
            elif self.is_cart:
                _set = self.tree.feature_sets[feat]
            # 然后调用这些二分标准并调用二类问题相关的计算信息量的方法
            if self.is_cart or self.wc[feat]:
                for tar in _set:
                    _tmp_gain, _tmp_chaos_list = _cluster.bin_info_gain(feat, tar, criterion=self.criterion,
                                                                        get_chaos_lst=True, continuous=self.wc[feat])
                    if _tmp_gain > _max_gain:
                        (_max_gain, _chaos_list), _max_feature, _max_tar = (_tmp_gain, _tmp_chaos_list), feat, tar
            # 对于离散型特征的ID3和C4.5算法，调用普通的计算信息量的方法
            else:
                _tmp_gain, _tmp_chaos_list = _cluster.info_gain(feat, self.criterion, True,
                                                                self.tree.feature_sets[feat])
                if _tmp_gain > _max_gain:
                    (_max_gain, _chaos_list), _max_feature = (_tmp_gain, _tmp_chaos_list), feat
        # 若满足第二种停止准则，则提出函数体
        if self.stop2(_max_gain, eps):
            return
        # 更新相关属性
        self.feature_dim = _max_feature
        if self.is_cart or self.wc[_max_feature]:
            self.tar = _max_tar
            # 调用根据划分标准进行生成的方法
            self._gen_children(_chaos_list)
            # 如果该node的左子节点和右子节点都是叶节点，且所属类别一致，那么久将他们合并，亦即进行局部剪枝
            if (self.left_child.category is not None and self.left_child.category == self.right_child.catefory):
                self.prune()
                # 调用tree的相关方法，将剪掉的该node的左右子节点从tree的记录所有node的列表nodes中除去
                self.tree.reduce_nodes()
        else:
            # 根据划分标准进行生成的方法
            self._gen_children(_chaos_list)

    # 子节点生成函数
    def _gen_children(self, chaos_lst, feature_bound):
        feat, tar = self.feature_dim, self.tar
        self.is_continuous = continuous = self.wc[feat]
        features = self._x[..., feat]
        new_feats = self.feats.copy()
        if continuous:
            mask = features < tar
            masks = [mask, ~mask]
        else:
            if self.is_cart:
                mask = features == tar
                masks = [mask, ~mask]
                self.tree.feature_sets[feat].discard(tar)
            else:
                masks = None
        if self.is_cart or continuous:
            feats = [tar, "+"] if not continuous else ["{:6.4}-".format(tar), "{:6.4}+".format(tar)]
            for feat, side, chaos in zip(feats, ["left_child", "right_child"], chaos_lst):
                new_node = self.__class__(
                    self.tree, self.base, chaos=chaos,
                    depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
                new_node.criterion = self.criterion
                setattr(self, side, new_node)
            for node, feat_mask in zip([self.left_child, self.right_child], masks):
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mask]
                    local_weights /= np.sum(local_weights)
                tmp_data, tmp_labels = self._x[feat_mask, ...], self._y[feat_mask]
                if len(tmp_labels) == 0:
                    continue
                node.feats = new_feats
                node.fit(tmp_data, tmp_labels, local_weights, feature_bound)
        else:
            new_feats.remove(self.feature_dim)
            for feat, chaos in zip(self.tree.feature_sets[self.feature_dim], chaos_lst):
                feat_mask = features == feat
                tmp_x = self._x[feat_mask, ...]
                if len(tmp_x) == 0:
                    continue
                new_node = self.__class__(
                    tree=self.tree, base=self.base, chaos=chaos,
                    depth=self._depth + 1, parent=self, is_root=False, prev_feat=feat)
                new_node.feats = new_feats
                self.children[feat] = new_node
                if self.sample_weight is None:
                    local_weights = None
                else:
                    local_weights = self.sample_weight[feat_mask]
                    local_weights /= np.sum(local_weights)
                new_node.fit(tmp_x, self._y[feat_mask], local_weights, feature_bound)

    def feed_tree(self, tree):
        self.tree = tree
        self.tree.nodes.append(self)
        self.wc = tree.whether_continuous
        for child in self.children.values():
            if child is not None:
                child.feed_tree(tree)

    def update_layers(self):
        # 根据Node的深度，在self.layers对应的位置列表中记录自己
        self.tree.layers[self._depth].append(self)
        # 遍历所有子节点，完成递归
        for _node in sorted(self.children):
            _node = self.children[_node]
            if _node is not None:
                _node.update_layers()

    def cost(self, pruned=False):
        if not pruned:
            return sum([leaf['chaos'] * len(leaf['y']) for leaf in self.leafs.values()])
        return self.chaos * len(self._y)


    def get_threshold(self):
        return (self.cost(pruned=True)-self.cost())/(len(self.leafs) -1)



if __name__ == '__main__':
    print('done!')
