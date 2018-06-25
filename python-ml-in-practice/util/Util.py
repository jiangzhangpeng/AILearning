# encoding:utf-8
import pickle
from math import ceil, sqrt

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


class VisUtil:
    @staticmethod
    def get_colors(line, all_positive):
        # c_base = 60
        # colors = []
        # for weight in line:
        #     colors.append([int(255 * (1 - weight)), int(255 - c_base * abs(1 - 2 * weight)), int(255 * weight)])
        # return colors
        # noinspection PyTypeChecker
        colors = np.full([len(line), 3], [0, 195, 255], dtype=np.uint8)
        if all_positive:
            return colors.tolist()
        colors[line < 0] = [255, 195, 0]
        return colors.tolist()

    @staticmethod
    def get_line_info(weight, max_thickness=4, threshold=0.2):
        w_min, w_max = np.min(weight), np.max(weight)
        if w_min >= 0:
            weight -= w_min
            all_pos = True
        else:
            all_pos = False
        weight /= max(w_max, -w_min)
        masks = np.abs(weight) >= threshold  # type: np.ndarray
        colors = [VisUtil.get_colors(line, all_pos) for line in weight]
        thicknesses = np.array(
            [[int((max_thickness - 1) * abs(n)) + 1 for n in line] for line in weight]
        )
        return colors, thicknesses, masks

    @staticmethod
    def get_graphs_from_logs():
        with open("Results/logs.dat", "rb") as file:
            logs = pickle.load(file)
        for (hus, ep, bt), log in logs.items():
            hus = list(map(lambda _c: str(_c), hus))
            title = "hus: {} ep: {} bt: {}".format(
                "- " + " -> ".join(hus) + " -", ep, bt
            )
            fb_log, acc_log = log["fb_log"], log["acc_log"]
            xs = np.arange(len(fb_log)) + 1
            plt.figure()
            plt.title(title)
            plt.plot(xs, fb_log)
            plt.plot(xs, acc_log, c="g")
            plt.savefig("Results/img/" + "{}_{}_{}".format(
                "-".join(hus), ep, bt
            ))
            plt.close()

    @staticmethod
    def show_img(img, title, normalize=True):
        if normalize:
            img_max, img_min = np.max(img), np.min(img)
            img = 255.0 * (img - img_min) / (img_max - img_min)
        plt.figure()
        plt.title(title)
        plt.imshow(img.astype('uint8'), cmap=plt.cm.gray)
        plt.gca().axis('off')
        plt.show()

    @staticmethod
    def show_batch_img(batch_img, title, normalize=True):
        _n, height, width = batch_img.shape
        a = int(ceil(sqrt(_n)))
        g = np.ones((a * height + a, a * width + a), batch_img.dtype)
        g *= np.min(batch_img)
        _i = 0
        for y in range(a):
            for x in range(a):
                if _i < _n:
                    g[y * height + y:(y + 1) * height + y, x * width + x:(x + 1) * width + x] = batch_img[_i, :, :]
                    _i += 1
        max_g = g.max()
        min_g = g.min()
        g = (g - min_g) / (max_g - min_g)
        VisUtil.show_img(g, title, normalize)

    @staticmethod
    def trans_img(img, shape=None):
        if shape is not None:
            img = img.reshape(shape)
        if img.shape[0] == 1:
            return img.reshape(img.shape[1:])
        return img.transpose(1, 2, 0)

    @staticmethod
    def make_mp4(ims, name="", fps=20, scale=1, extend=30):
        print("Making mp4...")
        ims += [ims[-1]] * extend
        with imageio.get_writer("{}.mp4".format(name), mode='I', fps=fps) as writer:
            for im in ims:
                if scale != 1:
                    new_shape = (int(im.shape[1] * scale), int(im.shape[0] * scale))
                    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
                    im = cv2.resize(im, new_shape, interpolation=interpolation)
                writer.append_data(im[..., ::-1])
        print("Done")
