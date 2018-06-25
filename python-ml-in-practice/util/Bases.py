# encoding:utf-8
import ctypes
import math
import multiprocessing
import sys

sys.path.append(r'D:\Workspaces\Github\AILearning\python-ml-in-practice\util')
import time
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from Timing import Timing
from Util import VisUtil


class ModelBase:
    """
        Base for all models
        Magic methods:
            1) __str__     : return self.name; __repr__ = __str__
            2) __getitem__ : access to protected members
        Properties:
            1) name  : name of this model, self.__class__.__name__ or self._name
            2) title : used in matplotlib (plt.title())
        Static method:
            1) disable_timing  : disable Timing()
            2) show_timing_log : show Timing() records
    """

    clf_timing = Timing()

    def __init__(self, **kwargs):
        self._plot_label_dict = {}
        self._title = self._name = None
        self._metrics, self._available_metrics = [], {
            "acc": ClassifierBase.acc
        }
        self._params = {
            "sample_weight": kwargs.get("sample_weight", None)
        }

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, "_" + item)

    @property
    def name(self):
        return self.__class__.__name__ if self._name is None else self._name

    @property
    def title(self):
        return str(self) if self._title is None else self._title

    @staticmethod
    def disable_timing():
        ModelBase.clf_timing.disable()

    @staticmethod
    def show_timing_log(level=2):
        ModelBase.clf_timing.show_timing_log(level)

    # Handle animation

    @staticmethod
    def _refresh_animation_params(animation_params):
        animation_params["show"] = animation_params.get("show", False)
        animation_params["mp4"] = animation_params.get("mp4", False)
        animation_params["period"] = animation_params.get("period", 1)

    def _get_animation_params(self, animation_params):
        if animation_params is None:
            animation_params = self._params["animation_params"]
        else:
            ClassifierBase._refresh_animation_params(animation_params)
        show, mp4, period = animation_params["show"], animation_params["mp4"], animation_params["period"]
        return show or mp4, show, mp4, period, animation_params

    def _handle_animation(self, i, x, y, ims, animation_params, draw_ani, show_ani, make_mp4, ani_period,
                          name=None, img=None):
        if draw_ani and x.shape[1] == 2 and (i + 1) % ani_period == 0:
            if img is None:
                img = self.get_2d_plot(x, y, **animation_params)
            if name is None:
                name = str(self)
            if show_ani:
                cv2.imshow(name, img)
                cv2.waitKey(1)
            if make_mp4:
                ims.append(img)

    def _handle_mp4(self, ims, animation_properties, name=None):
        if name is None:
            name = str(self)
        if animation_properties[2] and ims:
            VisUtil.make_mp4(ims, name)

    def get_2d_plot(self, x, y, padding=1, dense=200, draw_background=False, emphasize=None, extra=None, **kwargs):
        pass

    # Visualization

    def scatter2d(self, x, y, padding=0.5, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        indices = [labels == i for i in range(np.max(labels) + 1)]
        scatters = []
        plt.figure()
        plt.title(title)
        for idx in indices:
            scatters.append(plt.scatter(axis[0][idx], axis[1][idx], c=colors[idx]))
        plt.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                   ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def scatter3d(self, x, y, padding=0.1, title=None):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def transform_arr(arr):
            if arr.ndim == 1:
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        labels, n_label = transform_arr(labels)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]
        indices = [labels == i for i in range(n_label)]
        scatters = []
        fig = plt.figure()
        plt.title(title)
        ax = fig.add_subplot(111, projection='3d')
        for _index in indices:
            scatters.append(ax.scatter(axis[0][_index], axis[1][_index], axis[2][_index], c=colors[_index]))
        ax.legend(scatters, ["$c_{}$".format("{" + str(i) + "}") for i in range(len(scatters))],
                  ncol=math.ceil(math.sqrt(len(scatters))), fontsize=8)
        plt.show()

    # Util

    def predict(self, x, get_raw_results=False, **kwargs):
        pass


class ClassifierBase(ModelBase):
    """
        Base for classifiers
        Static methods:
            1) acc, f1_score           : Metrics
            2) _multi_clf, _multi_data : Parallelization
    """

    clf_timing = Timing()

    def __init__(self, **kwargs):
        super(ClassifierBase, self).__init__(**kwargs)
        self._params["animation_params"] = kwargs.get("animation_params", {})
        ClassifierBase._refresh_animation_params(self._params["animation_params"])

    # Metrics

    @staticmethod
    def acc(y, y_pred, weights=None):
        y, y_pred = np.asarray(y), np.asarray(y_pred)
        if weights is not None:
            return np.average((y == y_pred) * weights)
        return np.average(y == y_pred)

    # noinspection PyTypeChecker
    @staticmethod
    def f1_score(y, y_pred):
        tp = np.sum(y * y_pred)
        if tp == 0:
            return .0
        fp = np.sum((1 - y) * y_pred)
        fn = np.sum(y * (1 - y_pred))
        return 2 * tp / (2 * tp + fn + fp)

    # Parallelization

    # noinspection PyUnusedLocal
    @staticmethod
    def _multi_clf(x, clfs, task, kwargs, stack=np.vstack, target="single"):
        if target != "parallel":
            return np.array([clf.predict(x) for clf in clfs], dtype=np.float32).T
        n_cores = kwargs.get("n_cores", 2)
        n_cores = multiprocessing.cpu_count() if n_cores <= 0 else n_cores
        if n_cores == 1:
            matrix = np.array([clf.predict(x, n_cores=1) for clf in clfs], dtype=np.float32).T
        else:
            pool = Pool(processes=n_cores)
            batch_size = int(len(clfs) / n_cores)
            clfs = [clfs[i * batch_size:(i + 1) * batch_size] for i in range(n_cores)]
            x_size = np.prod(x.shape)  # type: int
            shared_base = multiprocessing.Array(ctypes.c_float, int(x_size))
            shared_matrix = np.ctypeslib.as_array(shared_base.get_obj()).reshape(x.shape)
            shared_matrix[:] = x
            matrix = stack(
                pool.map(task, ((shared_matrix, clfs, n_cores) for clfs in clfs))
            ).T.astype(np.float32)
        return matrix

    # noinspection PyUnusedLocal
    def _multi_data(self, x, task, kwargs, stack=np.hstack, target="single"):
        if target != "parallel":
            return task((x, self, 1))
        n_cores = kwargs.get("n_cores", 2)
        n_cores = multiprocessing.cpu_count() if n_cores <= 0 else n_cores
        if n_cores == 1:
            matrix = task((x, self, n_cores))
        else:
            pool = Pool(processes=n_cores)
            batch_size = int(len(x) / n_cores)
            batch_base, batch_data, cursor = [], [], 0
            x_dim = x.shape[1]
            for i in range(n_cores):
                if i == n_cores - 1:
                    batch_data.append(x[cursor:])
                    batch_base.append(multiprocessing.Array(ctypes.c_float, (len(x) - cursor) * x_dim))
                else:
                    batch_data.append(x[cursor:cursor + batch_size])
                    batch_base.append(multiprocessing.Array(ctypes.c_float, batch_size * x_dim))
                cursor += batch_size
            shared_arrays = [
                np.ctypeslib.as_array(shared_base.get_obj()).reshape(-1, x_dim)
                for shared_base in batch_base
            ]
            for i, data in enumerate(batch_data):
                shared_arrays[i][:] = data
            matrix = stack(
                pool.map(task, ((x, self, n_cores) for x in shared_arrays))
            )
        return matrix.astype(np.float32)

    # Training

    @staticmethod
    def _get_train_repeat(x, batch_size):
        train_len = len(x)
        batch_size = min(batch_size, train_len)
        do_random_batch = train_len > batch_size
        return 1 if not do_random_batch else int(train_len / batch_size) + 1

    def _batch_work(self, *args):
        pass

    def _batch_training(self, x, y, batch_size, train_repeat, *args):
        pass

    # Visualization

    def get_2d_plot(self, x, y, padding=1, dense=200, title=None,
                    draw_background=False, emphasize=None, extra=None, **kwargs):
        axis, labels = np.asarray(x).T, np.asarray(y)
        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])  # type: float
        y_min, y_max = np.min(axis[1]), np.max(axis[1])  # type: float
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)
        z = self.predict(base_matrix).reshape((nx, ny))

        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        buffer_ = io.BytesIO()
        plt.figure()
        if title is None:
            title = self.title
        plt.title(title)
        if draw_background:
            xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize)] = True
            plt.scatter(axis[0][indices], axis[1][indices], s=80,
                        facecolors="None", zorder=10)
        if extra is not None:
            plt.scatter(*np.asarray(extra).T, s=80, zorder=25, facecolors="red")

        plt.savefig(buffer_, format="png")
        plt.close()
        buffer_.seek(0)
        image = Image.open(buffer_)
        canvas = np.asarray(image)[..., :3]
        buffer_.close()
        return canvas

    def visualize2d(self, x, y, padding=0.1, dense=200, title=None,
                    show_org=False, draw_background=True, emphasize=None, extra=None, **kwargs):
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))
        nx, ny, padding = dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        def get_base(_nx, _ny):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            n_xf, n_yf = np.meshgrid(_xf, _yf)
            return _xf, _yf, np.c_[n_xf.ravel(), n_yf.ravel()]

        xf, yf, base_matrix = get_base(nx, ny)

        t = time.time()
        z = self.predict(base_matrix, **kwargs).reshape((nx, ny))
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        if labels.ndim == 1:
            if not self._plot_label_dict:
                self._plot_label_dict = {c: i for i, c in enumerate(set(labels))}
            dic = self._plot_label_dict
            n_label = len(dic)
            labels = np.array([dic[label] for label in labels])
        else:
            n_label = labels.shape[1]
            labels = np.argmax(labels, axis=1)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

        if title is None:
            title = self.title

        if show_org:
            plt.figure()
            plt.scatter(axis[0], axis[1], c=colors)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.show()

        plt.figure()
        plt.title(title)
        if draw_background:
            plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Pastel1)
        else:
            plt.contour(xf, yf, z, c='k-', levels=[0])
        plt.scatter(axis[0], axis[1], c=colors)
        if emphasize is not None:
            indices = np.array([False] * len(axis[0]))
            indices[np.asarray(emphasize)] = True
            plt.scatter(axis[0][indices], axis[1][indices], s=80,
                        facecolors="None", zorder=10)
        if extra is not None:
            plt.scatter(*np.asarray(extra).T, s=80, zorder=25, facecolors="red")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        print("Done.")

    def visualize3d(self, x, y, padding=0.1, dense=100, title=None,
                    show_org=False, draw_background=True, emphasize=None, extra=None, **kwargs):
        if False:
            print(Axes3D.add_artist)
        axis, labels = np.asarray(x).T, np.asarray(y)

        print("=" * 30 + "\n" + str(self))

        def decision_function(xx):
            return self.predict(xx, **kwargs)

        nx, ny, nz, padding = dense, dense, dense, padding
        x_min, x_max = np.min(axis[0]), np.max(axis[0])
        y_min, y_max = np.min(axis[1]), np.max(axis[1])
        z_min, z_max = np.min(axis[2]), np.max(axis[2])
        x_padding = max(abs(x_min), abs(x_max)) * padding
        y_padding = max(abs(y_min), abs(y_max)) * padding
        z_padding = max(abs(z_min), abs(z_max)) * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        z_min -= z_padding
        z_max += z_padding

        def get_base(_nx, _ny, _nz):
            _xf = np.linspace(x_min, x_max, _nx)
            _yf = np.linspace(y_min, y_max, _ny)
            _zf = np.linspace(z_min, z_max, _nz)
            n_xf, n_yf, n_zf = np.meshgrid(_xf, _yf, _zf)
            return _xf, _yf, _zf, np.c_[n_xf.ravel(), n_yf.ravel(), n_zf.ravel()]

        xf, yf, zf, base_matrix = get_base(nx, ny, nz)

        t = time.time()
        z_xyz = decision_function(base_matrix).reshape((nx, ny, nz))
        p_classes = decision_function(x).astype(np.int8)
        _, _, _, base_matrix = get_base(10, 10, 10)
        z_classes = decision_function(base_matrix).astype(np.int8)
        print("Decision Time: {:8.6} s".format(time.time() - t))

        print("Drawing figures...")
        z_xy = np.average(z_xyz, axis=2)
        z_yz = np.average(z_xyz, axis=1)
        z_xz = np.average(z_xyz, axis=0)

        xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
        yz_xf, yz_yf = np.meshgrid(yf, zf, sparse=True)
        xz_xf, xz_yf = np.meshgrid(xf, zf, sparse=True)

        def transform_arr(arr):
            if arr.ndim == 1:
                dic = {c: i for i, c in enumerate(set(arr))}
                n_dim = len(dic)
                arr = np.array([dic[label] for label in arr])
            else:
                n_dim = arr.shape[1]
                arr = np.argmax(arr, axis=1)
            return arr, n_dim

        labels, n_label = transform_arr(labels)
        p_classes, _ = transform_arr(p_classes)
        z_classes, _ = transform_arr(z_classes)
        colors = plt.cm.rainbow([i / n_label for i in range(n_label)])
        if extra is not None:
            ex0, ex1, ex2 = np.asarray(extra).T
        else:
            ex0 = ex1 = ex2 = None

        if title is None:
            try:
                title = self.title
            except AttributeError:
                title = str(self)

        if show_org:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(axis[0], axis[1], axis[2], c=colors[labels])
            plt.show()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        plt.title(title)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax1.set_title("Org")
        ax2.set_title("Pred")
        ax3.set_title("Boundary")

        ax1.scatter(axis[0], axis[1], axis[2], c=colors[labels])
        ax2.scatter(axis[0], axis[1], axis[2], c=colors[p_classes], s=15)
        if extra is not None:
            ax2.scatter(ex0, ex1, ex2, s=80, zorder=25, facecolors="red")
        xyz_xf, xyz_yf, xyz_zf = base_matrix[..., 0], base_matrix[..., 1], base_matrix[..., 2]
        ax3.scatter(xyz_xf, xyz_yf, xyz_zf, c=colors[z_classes], s=15)

        plt.show()
        plt.close()

        fig = plt.figure(figsize=(16, 4), dpi=100)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        def _draw(_ax, _x, _xf, _y, _yf, _z):
            if draw_background:
                _ax.pcolormesh(_x, _y, _z > 0, cmap=plt.cm.Pastel1)
            else:
                _ax.contour(_xf, _yf, _z, c='k-', levels=[0])

        def _emphasize(_ax, axis0, axis1, _c):
            _ax.scatter(axis0, axis1, c=_c)
            if emphasize is not None:
                indices = np.array([False] * len(axis[0]))
                indices[np.asarray(emphasize)] = True
                _ax.scatter(axis0[indices], axis1[indices], s=80,
                            facecolors="None", zorder=10)

        def _extra(_ax, axis0, axis1, _c, _ex0, _ex1):
            _emphasize(_ax, axis0, axis1, _c)
            if extra is not None:
                _ax.scatter(_ex0, _ex1, s=80, zorder=25, facecolors="red")

        colors = colors[labels]

        ax1.set_title("xy figure")
        _draw(ax1, xy_xf, xf, xy_yf, yf, z_xy)
        _extra(ax1, axis[0], axis[1], colors, ex0, ex1)

        ax2.set_title("yz figure")
        _draw(ax2, yz_xf, yf, yz_yf, zf, z_yz)
        _extra(ax2, axis[1], axis[2], colors, ex1, ex2)

        ax3.set_title("xz figure")
        _draw(ax3, xz_xf, xf, xz_yf, zf, z_xz)
        _extra(ax3, axis[0], axis[2], colors, ex0, ex2)

        plt.show()

        print("Done.")

    # Util

    def get_metrics(self, metrics):
        if len(metrics) == 0:
            for metric in self._metrics:
                metrics.append(metric)
            return metrics
        for i in range(len(metrics) - 1, -1, -1):
            metric = metrics[i]
            if isinstance(metric, str):
                try:
                    metrics[i] = self._available_metrics[metric]
                except AttributeError:
                    metrics.pop(i)
        return metrics

    @clf_timing.timeit(level=1, prefix="[API] ")
    def evaluate(self, x, y, metrics=None, tar=0, prefix="Acc", **kwargs):
        if metrics is None:
            metrics = ["acc"]
        self.get_metrics(metrics)
        logs, y_pred = [], self.predict(x, **kwargs)
        y = np.asarray(y)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        for metric in metrics:
            logs.append(metric(y, y_pred))
        if isinstance(tar, int):
            print(prefix + ": {:12.8}".format(logs[tar]))
        return logs


class KernelBase(ClassifierBase):
    """
    初始化结构
    self._fit_args,self._fit_args_names 记录循环体中所需要额外参数的信息的属性
    self._x ,self._y ,self._gram 记录数据集和gram矩阵的属性
    self._w ,self._b ,self._alpha 记录各种参数的属性
    self._kernel ,self._kernel_name ,self._kernel_param 记录核函数相关的属性
    self._prediction_cache ,self._dw_cache ,self._db_cache 记录y hat、dw、db的属性
    """

    def __init__(self):
        super(KernelBase, self).__init__()
        self._fit_args, self._fit_args_names = None, []
        self._x = self._y = self._gram = None
        self._w = self._b = self._alpha = None
        self._kernel = self._kernel_name = self._kernel_param = None
        self._prediction_cache = self._dw_cache = self._db_cache = None

    # 定义计算多项式矩阵的函数
    @staticmethod
    def _poly(x, y, p):
        return (x.dot(y.T) + 1) ** p

    # 定义计算RBF核矩阵的函数
    @staticmethod
    def _rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))

    # 默认是用RBF核，默认迭代次数epoch为1万次
    def fit(self, x, y, kernel='rbf', epoch=10 ** 4, **kwargs):
        self._x, self._y = np.atleast_2d(x), np.array(y)
        if kernel == 'poly':
            # 对于多项式核，默认使用kernelconfig中的default_p作为p的取值
            _p = kwargs.get('p', KernelConfig.default_p)
            self._kernel_name = 'Polynomial'
            self._kernel_param = 'degree = {}'.format(_p)
            self._kernel = lambda _x, _y: KernelBase._poly(_x, _y, _p)
        elif kernel == 'rbf':
            # 对于rbf核，默认使用样本x的位数你的倒数1\n作为gamma取值
            _gamma = kwargs.get('gamma', 1 / self._x.shape[1])
            self._kernel_name = 'RBF'
            self._kernel_param = 'gamma = {}'.format(_gamma)
            self._kernel = lambda _x, _y: KernelBase._rbf(_x, _y, _gamma)

        # 初始化参数
        self._alpha, self._w, self._prediction_cache = (np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)))
        #通过核方法计算出gram矩阵
        self._gram = self._kernel(self._x,self._x)
        self._b = 0
        #调用_prepare方法进行特殊参数的初始化（比如SVM中的惩罚因子C）
        self._prepare(**kwargs)
        #获取在循环体中会用到的参数
        _fit_args = []
        for _name,_arg in zip(self._fit_args_names,self._fit_args):
            if _name in kwargs:
                _arg = kwargs[_name]
            _fit_args.append(_arg)
        #迭代，直到达到迭代次数或_fit核心方法返回真值
        for _ in range(epoch):
            if self._fit(sample_weight,*_fit_args):
                break
        #利用alpha和训练样本来更新w和b
        self._updata_params()

    #定义更新预测向量_prediction_cache的函数
    def _update_pred_cache(self,*args):
        self._prediction_cache += self._db_cache
        if len(args) == 1:
            self._prediction_cache += self._dw_cache * self._gram[args[0]]
        else:
            self._prediction_cache += self._dw_cache.dot(self._gram[args,...])

    #定义预测函数
    def predict(self,x,get_raw_result = False):
        #计算测试集和训练集之间的核函数矩阵并利用它来做决策
        x = self._kernel(np.atleast_2d(x),self._x)
        y_pred = x.dot(self._w) + self._b
        if not get_raw_result:
            return np.sign(y_pred)
        return y_pred
    






















class KernelConfig:
    default_p = 3
    default_c = 1
