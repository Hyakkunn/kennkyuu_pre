import os
from glob import glob
import threading
import multiprocessing
import signal
import sys
from datetime import datetime

import tensorflow as tf
import numpy as np
import cairosvg
from PIL import Image
import io
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import random
import cv2

from ops import *
from skimage.morphology import skeletonize

# 画像を生成する際のsvgテンプレート
# テンプレート名 = 描く線分の情報(詳しくは，http://defghi1977.html.xdomain.jp/tech/svgMemo/svgMemo_03.htm または，https://qiita.com/takeshisakuma/items/777e3cb0a54ea7b1dbe7 を参照)

SVG_START_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{w}" height="{h}" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg" version="1.1">
<g fill="none">\n"""
SVG_LINE_TEMPLATE = """<path id="{id}" d="M {x1} {y1} L{x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""
SVG_LINE_dash_TEMPLATE = """<path id="{id}" d="M {x1} {y1} L{x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1},{sd2}"/>"""
SVG_LINE_dash1_TEMPLATE = """<path id="{id}" d="M {x1} {y1} L{x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1} {sde} {sdf} {sde}"/>"""
SVG_LINE_dash2_TEMPLATE = """<path id="{id}" d="M {x1} {y1} L{x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1} {sde} {sdf} {sde} {sdf} {sde}"/>"""

SVG_CUBIC_BEZIER_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""
SVG_CUBIC_BEZIER_dash_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1},{sd2}"/>"""
SVG_CUBIC_BEZIER_dash1_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1} {sde} {sdf} {sde}"/>"""
SVG_CUBIC_BEZIER_dash2_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1} {sde} {sdf} {sde} {sdf} {sde}"/>"""

SVG_QUADRATIC_TEMPLATE = """<path id="{id}" d="M {x1},{y1} m 0,0 q {q1},{q2} {q3},{q4}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""
SVG_QUADRATIC_dash_TEMPLATE = """<path id="{id}" d="M {x1},{y1} m 0,0 q {q1},{q2} {q3},{q4}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1},{sd2}"/>"""
SVG_QUADRATIC_dash1_TEMPLATE = """<path id="{id}" d="M {x1},{y1} m 0,0 q {q1},{q2} {q3},{q4}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1} {sde} {sdf} {sde}"/>"""
SVG_QUADRATIC_dash2_TEMPLATE = """<path id="{id}" d="M {x1},{y1} m 0,0 q {q1},{q2} {q3},{q4}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1} {sde} {sdf} {sde} {sdf} {sde}"/>"""

SVG_CUBIC_FUNCTION_TEMPLATE = """<path id="{id}" d="M {x1} {y1} C {vpx1} {vpy1}, {vpx2} {vpy1}, {cp1} {cp2} C {vpx3} {vpy2}, {vpx4} {vpy2}, {x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""
SVG_CUBIC_FUNCTION_dash_TEMPLATE = """<path id="{id}" d="M {x1} {y1} C {vpx1} {vpy1}, {vpx2} {vpy1}, {cp1} {cp2} C {vpx3} {vpy2}, {vpx4} {vpy2}, {x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1},{sd2}"/>"""
SVG_CUBIC_FUNCTION_dash1_TEMPLATE = """<path id="{id}" d="M {x1} {y1} C {vpx1} {vpy1}, {vpx2} {vpy1}, {cp1} {cp2} C {vpx3} {vpy2}, {vpx4} {vpy2}, {x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1} {sde} {sdf} {sde}"/>"""
SVG_CUBIC_FUNCTION_dash2_TEMPLATE = """<path id="{id}" d="M {x1} {y1} C {vpx1} {vpy1}, {vpx2} {vpy1}, {cp1} {cp2} C {vpx3} {vpy2}, {vpx4} {vpy2}, {x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1} {sde} {sdf} {sde} {sdf} {sde}"/>"""

# SVG_CIRCLE_TEMPLATE = """<circle id="{id}" cx="{cx}" cy="{cy}" r="{ra}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""
# SVG_CIRCLE_dash_TEMPLATE = """<circle id="{id}" cx="{cx}" cy="{cy}" r="{ra}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1},{sd2}"/>"""

# SVG_CIRCLE_TEMPLATE = """<path id="{id}" d="M {x1} {y1} A {rx} {ry} 0 1 0 {x2} {y2} M {x2} {y2} A {rx} {ry} 0 1 0 {x1} {y1}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""
# SVG_CIRCLE_dash_TEMPLATE = """<path id="{id}" d="M {x1} {y1} A {rx} {ry} 0 1 0 {x2} {y2} M {x2} {y2} A {rx} {ry} 0 1 0 {x1} {y1}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1},{sd2}"/>"""

# SVG_RECT_TEMPLATE = """<rect id="{id}" x="{x}" y="{y}" width="{w}" height="{h}" stroke="rgb({r},{g},{b})" stroke-width="{sw}"/>"""
# SVG_RECT_dash_TEMPLATE = """<rect id="{id}" x="{x}" y="{y}" width="{w}" height="{h}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" stroke-dasharray="{sd1},{sd2}"/>"""

SVG_END_TEMPLATE = """</g>\n</svg>"""


class BatchManager(object):
    def __init__(self, config):
        self.root = config.data_path
        self.rng = np.random.RandomState(config.random_seed)

        self.paths = sorted(glob("{}/train/*.{}".format(self.root, 'svg_pre'))) # 実際に学習やテストに使用するのは「〇〇.svg_pre」で「〇〇.svg」ではない．
        if len(self.paths) == 0:
            # create line dataset
            data_dir = os.path.join(config.data_dir, config.dataset)
            train_dir = os.path.join(data_dir, 'train')
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            test_dir = os.path.join(data_dir, 'test')
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            self.paths = gen_data(data_dir, config, self.rng,
                                  num_train=500, num_test=100) # 画像枚数(学習用，テスト(評価)用)

        self.test_paths = sorted(glob("{}/test/*.{}".format(self.root, 'svg_pre')))
        assert (len(self.paths) > 0 and len(self.test_paths) > 0)

        self.batch_size = config.batch_size
        self.height = config.height
        self.width = config.width

        self.is_pathnet = (config.archi == 'path')
        if self.is_pathnet:
            feature_dim = [self.height, self.width, 2]
            label_dim = [self.height, self.width, 1]
        else:
            feature_dim = [self.height, self.width, 1]
            label_dim = [self.height, self.width, 1]

        self.capacity = 10000
        self.q = tf.FIFOQueue(self.capacity, [tf.float32, tf.float32], [feature_dim, label_dim])
        self.x = tf.placeholder(dtype=tf.float32, shape=feature_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=label_dim)
        self.enqueue = self.q.enqueue([self.x, self.y])
        self.num_threads = config.num_worker
        self.is_binary = config.is_binary # data_line_2
        # np.amin([config.num_worker, multiprocessing.cpu_count(), self.batch_size])

    def __del__(self):
        try:
            self.stop_thread()
        except AttributeError:
            pass

    def start_thread(self, sess):
        print('%s: start to enque with %d threads' % (datetime.now(), self.num_threads))

        # Main thread: create a coordinator.
        self.sess = sess
        self.coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, paths, rng,
                           x, y, w, h, is_pathnet):
            with coord.stop_on_exception():
                while not coord.should_stop():
                    id = rng.randint(len(paths))
                    if is_pathnet:
                        x_, y_ = preprocess_path(paths[id], w, h, rng, self.is_binary)
                    else:
                        x_, y_ = preprocess_overlap(paths[id], w, h, rng, self.is_binary)
                    sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self.threads = [threading.Thread(target=load_n_enqueue,
                                         args=(self.sess,
                                               self.enqueue,
                                               self.coord,
                                               self.paths,
                                               self.rng,
                                               self.x,
                                               self.y,
                                               self.width,
                                               self.height,
                                               self.is_pathnet)
                                         ) for i in range(self.num_threads)]

        # define signal handler
        def signal_handler(signum, frame):
            # print "stop training, save checkpoint..."
            # saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
            print('%s: canceled by SIGINT' % datetime.now())
            self.coord.request_stop()
            self.sess.run(self.q.close(cancel_pending_enqueues=True))
            self.coord.join(self.threads)
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self.threads:
            t.start()

        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False
        qs = 0
        while qs < (self.capacity * 0.8):
            qs = self.sess.run(self.q.size())
            print(qs)
        print('%s: q size %d' % (datetime.now(), qs))

    def stop_thread(self):
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

        self.coord.request_stop()
        self.sess.run(self.q.close(cancel_pending_enqueues=True))
        self.coord.join(self.threads)

    def test_batch(self):
        x_list, y_list = [], []
        for i, file_path in enumerate(self.test_paths):
            if self.is_pathnet:
                # x_, y_ = preprocess_path(file_path, self.width, self.height, self.rng)
                x_, y_ = preprocess_path(file_path, self.width, self.height, self.rng, self.is_binary) # data_line_2
            else:
                # x_, y_ = preprocess_overlap(file_path, self.width, self.height, self.rng)
                x_, y_ = preprocess_overlap(file_path, self.width, self.height, self.rng, self.is_binary) # data_line_2
            x_list.append(x_)
            y_list.append(y_)
            if i % self.batch_size == self.batch_size - 1:
                yield np.array(x_list), np.array(y_list)
                x_list, y_list = [], []

    def batch(self):
        return self.q.dequeue_many(self.batch_size)

    def sample(self, num):
        idx = self.rng.choice(len(self.paths), num).tolist()
        return [self.paths[i] for i in idx]

    def random_list(self, num):
        x_list = []
        xs, ys = [], []
        file_list = self.sample(num)
        for file_path in file_list:
            if self.is_pathnet:
                # x, y = preprocess_path(file_path, self.width, self.height, self.rng)
                x, y = preprocess_path(file_path, self.width, self.height, self.rng, self.is_binary)
            else:
                # x, y = preprocess_overlap(file_path, self.width, self.height, self.rng)
                x, y = preprocess_overlap(file_path, self.width, self.height, self.rng, self.is_binary)
            x_list.append(x)

            if self.is_pathnet:
                b_ch = np.zeros([self.height, self.width, 1])
                xs.append(np.concatenate((x * 255, b_ch), axis=-1))
            else:
                xs.append(x * 255)
            ys.append(y * 255)

        return np.array(x_list), np.array(xs), np.array(ys), file_list

    def read_svg(self, file_path): # 学習用および評価用画像の作成
        with open(file_path, 'r') as f:
            svg = f.read()

        # svg = svg.format(w=self.width, h=self.height)
        svg_xml = et.fromstring(svg)
        svg_xml.attrib["height"] = "512"
        svg_xml.attrib["width"] = "512"
        svg = et.tostring(svg_xml, method="xml")
        img = cairosvg.svg2png(bytestring=svg)
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > 0) * max_intensity #2値化
        s = s / max_intensity
        # # np.save('svg_array', s)

        path_list = []
        svg_xml = et.fromstring(svg)
        num_paths = len(svg_xml[0])

        for i in range(num_paths):
            svg_xml = et.fromstring(svg)
            svg_xml[0][0] = svg_xml[0][i]
            del svg_xml[0][1:]  # [0][2:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            path = (np.array(y_img)[:, :, 3] > 0)
            if self.is_binary:
                path = (path > 0) * max_intensity #二値化
            path_list.append(path)

        return s, num_paths, path_list

    def read_svg_for_DtoS(self, file_path, height, width): # モデルの評価に使用する画像の読み込みと編集 # data\testを読み込む
        with open(file_path, 'r') as f:
            svg = f.read()

        # svg = svg.format(w=self.width, h=self.height)
        svg_xml = et.fromstring(svg)
        svg_xml_dash = et.fromstring(svg)
        # svg_xml_sketch = et.fromstring(svg) # sketch
        svg_xml.attrib["height"] = "512"
        svg_xml.attrib["width"] = "512"
        svg_xml_dash.attrib["height"] = "512"
        svg_xml_dash.attrib["width"] = "512"
        # svg_xml_sketch.attrib["height"] = "800" # sketch
        # svg_xml_sketch.attrib["width"] = "800" # sketch
#
        # # sketchの正解画像の作成(前段階)
        # svg_sketch_gt = et.tostring(svg_xml_sketch, method="xml")
        # png_sketch_gt = cairosvg.svg2png(bytestring=svg_sketch_gt)
        # img_sketch_gt = Image.open(io.BytesIO(png_sketch_gt))
        # gt_sketch = np.array(img_sketch_gt)[:, :, 3].astype(np.float)
        # max_intensity_sketch = np.amax(gt_sketch)
        # gt_sketch = (gt_sketch > 0) * max_intensity_sketch
        # gt_sketch = gt_sketch / max_intensity_sketch
        # gt_sketch_sk = skeletonize(gt_sketch)
        # gt_sketch_sk = gt_sketch_sk.astype(np.float)
        # gt_sketch_sk_3 = gt_sketch_sk.astype(np.float)
        # gt_f_sketch = gt_sketch_sk - gt_sketch
        # gt_f_sketch = (gt_f_sketch > 0) * 1.0
        # gt_sketch_sk = gt_sketch_sk - gt_f_sketch
#
        # # sketchの破線画像の作成
        # stroke_width = svg_xml_sketch[0].attrib["stroke-width"]
        # stroke_width = float(stroke_width)
        # scale = svg_xml_sketch[0][0].attrib["transform"]
        # scale = float(scale.split()[1][6:-1])
        # while True:
        #     sd1 = (stroke_width * random.randrange(5, 13, 1) / 2)
        #     sd2 = (3 * random.randrange(2, 10, 1) / 2)
        #     if sd1 < sd2:
        #         continue
        #     break
#
        # svg_xml_sketch[0].attrib["stroke-dasharray"] = "{},{}".format(sd1, sd2)
#
        # # sketchの入力画像の作成
        # svg_sketch = et.tostring(svg_xml_sketch, method="xml")
        # img_sketch = cairosvg.svg2png(bytestring=svg_sketch)
        # img_sketch = Image.open(io.BytesIO(img_sketch))
        # s_sketch = np.array(img_sketch)[:, :, 3].astype(np.float)
        # s_sketch = (s_sketch > 0) * max_intensity_sketch
        # img_sketch = s_sketch / max_intensity_sketch
#
        # # sketchの正解画像の作成
        # gt_sketch_sk = gt_sketch_sk - img_sketch
        # gt_sketch_sk = (gt_sketch_sk > 0) * 1.0
#
        # # sketch 出力の値の代入
        # img = img_sketch
        # img2 = img_sketch
        # num_paths = 0
        # gt = gt_sketch_sk
        # y = gt_sketch
        # gt_3 = gt_sketch_sk_3

        # 入力画像の二値化
        svg = et.tostring(svg_xml, method="xml")
        img = cairosvg.svg2png(bytestring=svg)
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
        # s = np.array(img)[:, :, 3]
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > 0) * max_intensity #2値化
            # s = cv2.GaussianBlur(s, (5, 5), 0)
            # s = cv2.threshold(s, 0, max_intensity, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # s = s[1]
        img = s / max_intensity
        # # np.save('svg_array', s)
        path_list = []
        #svg_xml = et.fromstring(svg)
        num_paths = len(svg_xml[0]) # tikz_graph → svg_xml[1]
        tikz_svg = 0
        if num_paths == 1:
            num_paths = len(svg_xml[1])
            tikz_svg = 1
        #svg_xml.attrib["height"] = "512"
        #svg_xml.attrib["width"] = "512"
        # 入力画像の作成(破線のみ)
        for i in range(0, num_paths):
            if tikz_svg == 0:
                if "stroke-dasharray" in svg_xml_dash[0][i].attrib.keys():
                    continue
                else:
                    del svg_xml_dash[0][i].attrib["stroke"]
            else:
                if "stroke-dasharray" in svg_xml_dash[1][i].attrib.keys():
                    continue
                else:
                    del svg_xml_dash[1][i].attrib["stroke"]
        svg2 = et.tostring(svg_xml_dash, method="xml")
        img2 = cairosvg.svg2png(bytestring=svg2)
        img2 = Image.open(io.BytesIO(img2))
        s2 = np.array(img2)[:, :, 3].astype(np.float)  # / 255.0
        # s2 = np.array(img2)[:, :, 3]
        max_intensity2 = np.amax(s2)
        if self.is_binary:
            s2 = (s2 > 0) * max_intensity2  # 2値化 # data_line_2
            # s2 = cv2.GaussianBlur(s2, (5, 5), 0)
            # s2 = cv2.threshold(s2, 0, max_intensity, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # s2 = s2[1]
        img2 = s2 / max_intensity2

        # 正解画像の作成(実線あり)
        for i in range(0, num_paths):  # data_line_2 st
            if tikz_svg == 0:
                if "stroke-dasharray" in svg_xml[0][i].attrib.keys():  # 破線を実線に変える
                    del svg_xml[0][i].attrib["stroke-dasharray"]
            else:
                if "stroke-dasharray" in svg_xml[1][i].attrib.keys():  # 破線を実線に変える
                    del svg_xml[1][i].attrib["stroke-dasharray"]
        svg_str = et.tostring(svg_xml, method='xml')

        # # 正解画像の作成(破線のみ)
        # for i in range(0, num_paths):  # data_line_2 st
        #     if tikz_svg == 0:
        #         if "stroke-dasharray" in svg_xml[0][i].attrib.keys():  # 破線を実線に変える
        #             del svg_xml[0][i].attrib["stroke-dasharray"]
        #         else:
        #             del svg_xml[0][i].attrib["stroke"]
        #     else:
        #         if "stroke-dasharray" in svg_xml[1][i].attrib.keys():  # 破線を実線に変える
        #             del svg_xml[1][i].attrib["stroke-dasharray"]
        #         else:
        #             del svg_xml[1][i].attrib["stroke"]
        # svg_str = et.tostring(svg_xml, method='xml')

        # leave only one path
        y_png = cairosvg.svg2png(bytestring=svg_str)
        y_img = Image.open(io.BytesIO(y_png))
        # y_img = skeletonize(y_img)
        # y = np.array(y_img)[:, :, 3].astype(np.float) / max_intensity
        # y = np.reshape(y, [h, w, 1])
        y = np.array(y_img)[:, :, 3].astype(np.float)  # [0,1]
        # y = np.array(y_img)[:, :, 3]
        y = (y > 0) * max_intensity  # 2値化
        # y = cv2.GaussianBlur(y, (5, 5), 0)
        # y = cv2.threshold(y, 0, max_intensity, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # y = y[1]
        y = y / max_intensity
        gt = skeletonize(y)
        gt = gt.astype(np.float)
        gt_3 = gt.astype(np.float)
        gt_f = gt - y
        gt_f = (gt_f > 0) * 1.0
        gt = gt - gt_f
        gt = gt - img # img(実線込み) img2(破線のみ)
        gt = (gt > 0) * 1.0
        # gt = np.reshape(y, [height, width, 1])
        # for i in range(num_paths):
        #     svg_xml = et.fromstring(svg)
        #     svg_xml[0][0] = svg_xml[0][i]
        #     del svg_xml[0][1:]  # [0][2:] # dtaa_line_2
        #     svg_one = et.tostring(svg_xml, method='xml')
        #     # leave only one path
        #     y_png = cairosvg.svg2png(bytestring=svg_one)
        #     y_img = Image.open(io.BytesIO(y_png))
        #     path = (np.array(y_img)[:, :, 3] > 0)
        #     if self.is_binary:
        #         path = (path > 0) * max_intensity #二値化 # dtaa_line_2
        #     path_list.append(path)
        # img2 は破線のみの入力画像が必要な時に使用
        return img, num_paths, gt, img2, y, gt_3 #img=入力画像、num_paths=線分の数、gt=正解画像から破線ピクセルを抜いた画像,img2=破線のみの画像,y=正解画像,gt_3=

# これ以降は作成する学習画像の関数
def draw_line(id, w, h, min_length, max_stroke_width, rng): # 実線の直線
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = rng.uniform(low=1, high=max_stroke_width + 1)
    # stroke_width = max_stroke_width
    while True:
        x = rng.randint(w, size=2)
        y = rng.randint(h, size=2)
        if x[0] - x[1] + y[0] - y[1] < min_length:
            continue
        break

    return SVG_LINE_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        x2=x[1], y2=y[1],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width
    )


# def draw_line_contact(id, w, h, min_length, max_stroke_width, rng, co):
#     stroke_color = rng.randint(240, size=3)
#     stroke_color = [255, 255, 0]
#     # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
#     stroke_width = max_stroke_width
#     while True:
#         x = rng.randint(w, size=2)
#         y = rng.randint(h, size=2)
#         if x[0] == co[0] or x[1] < co[0] or x[0] > co[0]:
#             continue
#         a = (y[0] - co[1]) / (x[0] - co[0])
#         b = y[0] - (a * x[0])
#         f = (a * x[1]) + b
#         if f < 0 or f > 64:
#             continue
#         break
#
#     return SVG_LINE_TEMPLATE.format(
#         id=id,
#         x1=x[0], y1=y[0],
#         x2=x[1], y2=f,
#         r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
#         sw=stroke_width
#     )


# def draw_quadratic_contact(id, w, h, min_length, max_stroke_width, rng, co):
#     stroke_color = rng.randint(240, size=3)
#     # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
#     stroke_width = max_stroke_width
#     while True:
#         x = rng.randint(w, size=1)
#         y = rng.randint(h, size=1)
#         if x[0] > co[0] or y[0] == co[1]:
#             continue
#         break
#
#     q1 = co[0] - x[0]
#     q2 = (co[1] - y[0]) * 2
#     q3 = q1 * 2
#
#     return SVG_QUADRATIC_TEMPLATE.format(
#         id=id,
#         x1=x[0], y1=y[0],
#         q1=q1, q2=q2, q3=q3, q4=0,
#         r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
#         sw=stroke_width
#     )


def draw_line_dash(id, w, h, min_length, max_stroke_width, rng): # 点線の直線
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = rng.uniform(low=1, high=max_stroke_width + 1)
    # stroke_width = max_stroke_width
    while True:
        x = rng.randint(w, size=2)
        y = rng.randint(h, size=2)
        if x[0] - x[1] + y[0] - y[1] < min_length:
            continue
        break

    while True:
        sd1 = stroke_width * random.randrange(5, 13, 1) / 2
        sd2 = stroke_width * random.randrange(2, 10, 1) / 2
        if sd1 < sd2:
            continue
        break

    return SVG_LINE_dash_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        x2=x[1], y2=y[1],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=sd1,
        sd2=sd2
    )

def draw_line_dash1(id, w, h, min_length, max_stroke_width, rng): # 一点鎖線の直線
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = rng.uniform(low=1, high=max_stroke_width + 1)
    # stroke_width = max_stroke_width
    while True:
        x = rng.randint(w, size=2)
        y = rng.randint(h, size=2)
        if x[0] - x[1] + y[0] - y[1] < min_length:
            continue
        break

    sd1 = stroke_width * random.randrange(5, 13, 1) / 2
    sde = stroke_width * random.randrange(3, 5, 1) / 2
    sdf = stroke_width * random.randrange(2, 4, 1) / 2

    return SVG_LINE_dash1_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        x2=x[1], y2=y[1],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=sd1,
        sde=sde,
        sdf=sdf
    )

def draw_line_dash2(id, w, h, min_length, max_stroke_width, rng): # 二点鎖線の直線
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = rng.uniform(low=1, high=max_stroke_width + 1)
    # stroke_width = max_stroke_width
    while True:
        x = rng.randint(w, size=2)
        y = rng.randint(h, size=2)
        if x[0] - x[1] + y[0] - y[1] < min_length:
            continue
        break

    sd1 = stroke_width * random.randrange(5, 13, 1) / 2
    sde = stroke_width * random.randrange(3, 5, 1) / 2
    sdf = stroke_width * random.randrange(2, 4, 1) / 2

    return SVG_LINE_dash2_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        x2=x[1], y2=y[1],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=sd1,
        sde=sde,
        sdf=sdf
    )

def draw_cubic_bezier_curve(id, w, h, min_length, max_stroke_width, rng): # 実線の三次ベジエ曲線
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = rng.uniform(low=1, high=max_stroke_width + 1)
    # stroke_width = max_stroke_width
    x = rng.randint(w, size=4)
    y = rng.randint(h, size=4)

    return SVG_CUBIC_BEZIER_TEMPLATE.format(
        id=id,
        sx=x[0], sy=y[0],
        cx1=x[1], cy1=y[1],
        cx2=x[2], cy2=y[2],
        tx=x[3], ty=y[3],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width
    )


def draw_cubic_bezier_curve_dash(id, w, h, min_length, max_stroke_width, rng): # 点線の三次ベジエ曲線
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = rng.uniform(low=1, high=max_stroke_width + 1)
    # stroke_width = max_stroke_width
    x = rng.randint(w, size=4)
    y = rng.randint(h, size=4)

    while True:
        sd1 = stroke_width * random.randrange(5, 13, 1) / 2
        sd2 = stroke_width * random.randrange(2, 10, 1) / 2
        if sd1 < sd2:
            continue
        break

    return SVG_CUBIC_BEZIER_dash_TEMPLATE.format(
        id=id,
        sx=x[0], sy=y[0],
        cx1=x[1], cy1=y[1],
        cx2=x[2], cy2=y[2],
        tx=x[3], ty=y[3],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=sd1,
        sd2=sd2
    )

def draw_cubic_bezier_curve_dash1(id, w, h, min_length, max_stroke_width, rng): # 一点鎖線の三次ベジエ曲線
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = rng.uniform(low=1, high=max_stroke_width + 1)
    # stroke_width = max_stroke_width
    x = rng.randint(w, size=4)
    y = rng.randint(h, size=4)

    sd1 = stroke_width * random.randrange(5, 13, 1) / 2
    sde = stroke_width * random.randrange(3, 5, 1) / 2
    sdf = stroke_width * random.randrange(2, 4, 1) / 2

    return SVG_CUBIC_BEZIER_dash1_TEMPLATE.format(
        id=id,
        sx=x[0], sy=y[0],
        cx1=x[1], cy1=y[1],
        cx2=x[2], cy2=y[2],
        tx=x[3], ty=y[3],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=sd1,
        sde=sde,
        sdf=sdf
    )

def draw_cubic_bezier_curve_dash2(id, w, h, min_length, max_stroke_width, rng): # 二点鎖線の三次ベジエ曲線
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = rng.uniform(low=1, high=max_stroke_width + 1)
    # stroke_width = max_stroke_width
    x = rng.randint(w, size=4)
    y = rng.randint(h, size=4)

    sd1 = stroke_width * random.randrange(5, 13, 1) / 2
    sde = stroke_width * random.randrange(3, 5, 1) / 2
    sdf = stroke_width * random.randrange(2, 4, 1) / 2

    return SVG_CUBIC_BEZIER_dash2_TEMPLATE.format(
        id=id,
        sx=x[0], sy=y[0],
        cx1=x[1], cy1=y[1],
        cx2=x[2], cy2=y[2],
        tx=x[3], ty=y[3],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=sd1,
        sde=sde,
        sdf=sdf
    )

def draw_quadratic_function(id, w, h, min_length, max_stroke_width, rng): # 実線の二次関数
    stroke_color = rng.randint(240, size=3)
    stroke_width = rng.uniform(low=1, high=max_stroke_width+1)
    # stroke_width = max_stroke_width

    while True:
        cx = rng.randint(150, 250, size=1)
        cy = rng.randint(h, size=1)
        x = rng.randint(w, size=1)
        y = rng.randint(h, size=1)
        if x[0] >= cx[0] or ((-50 <= (cy[0] - y[0])) and ((cy[0] - y[0]) <= 50)):
            continue
        break

    q1 = cx[0] - x[0]
    q2 = (cy[0] - y[0]) * 2
    q3 = q1 * 2

    return SVG_QUADRATIC_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        q1=q1, q2=q2, q3=q3, q4=0,
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width
    )

def draw_quadratic_dash_function(id, w, h, min_length, max_stroke_width, rng): # 点線の二次関数
    stroke_color = rng.randint(240, size=3)
    stroke_width = rng.uniform(low=1, high=max_stroke_width+1)
    # stroke_width = max_stroke_width

    while True:
        cx = rng.randint(150, 250, size=1)
        cy = rng.randint(h, size=1)
        x = rng.randint(w, size=1)
        y = rng.randint(h, size=1)
        if x[0] >= cx[0] or ((-50 <= (cy[0] - y[0])) and ((cy[0] - y[0]) <= 50)):
            continue
        break

    q1 = cx[0] - x[0]
    q2 = (cy[0] - y[0]) * 2
    q3 = q1 * 2

    while True:
        sd1 = stroke_width * random.randrange(5, 13, 1) / 2
        sd2 = stroke_width * random.randrange(2, 10, 1) / 2
        if sd1 < sd2:
            continue
        break

    return SVG_QUADRATIC_dash_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        q1=q1, q2=q2, q3=q3, q4=0,
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=sd1, sd2=sd2
    )

def draw_quadratic_dash12_function(id, w, h, min_length, max_stroke_width, rng): # 鎖線の二次関数
    stroke_color = rng.randint(240, size=3)
    stroke_width = rng.uniform(low=1, high=max_stroke_width+1)
    # stroke_width = max_stroke_width

    while True:
        cx = rng.randint(150, 250, size=1)
        cy = rng.randint(h, size=1)
        x = rng.randint(w, size=1)
        y = rng.randint(h, size=1)
        if x[0] >= cx[0] or ((-50 <= (cy[0] - y[0])) and ((cy[0] - y[0]) <= 50)):
            continue
        break

    q1 = cx[0] - x[0]
    q2 = (cy[0] - y[0]) * 2
    q3 = q1 * 2

    sd1 = stroke_width * random.randrange(5, 13, 1) / 2
    sde = stroke_width * random.randrange(3, 5, 1) / 2
    sdf = stroke_width * random.randrange(2, 4, 1) / 2

    dc = rng.randint(0, 2)

    if dc == 0:
        return SVG_QUADRATIC_dash1_TEMPLATE.format(
            id=id,
            x1=x[0], y1=y[0],
            q1=q1, q2=q2, q3=q3, q4=0,
            r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
            sw=stroke_width,
            sd1=sd1, sde=sde, sdf=sdf
        )
    else:
        return SVG_QUADRATIC_dash2_TEMPLATE.format(
            id=id,
            x1=x[0], y1=y[0],
            q1=q1, q2=q2, q3=q3, q4=0,
            r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
            sw=stroke_width,
            sd1=sd1, sde=sde, sdf=sdf
        )

def draw_cubic_function(id, w, h, min_length, max_stroke_width, rng): # 実線の三次関数
    stroke_color = rng.randint(240, size=3)
    stroke_width = rng.uniform(low=1, high=max_stroke_width+1)

    x1 = rng.randint(0, 50)
    y = rng.randint(100, 300, size=2)
    cp = rng.randint(150, 250, size=2)
    x2 = (2 * cp[0]) - x1

    vpx1 = x1 + ((cp[0] - x1) / 3)
    vpx2 = vpx1 + ((cp[0] - x1) / 3)
    vpx3 = cp[0] + ((cp[0] - x1) / 3)
    vpx4 = vpx3 + ((cp[0] - x1) / 3)

    while True:
        vpy1 = rng.randint(h)
        if (-30 <= (cp[1] - vpy1)) and ((cp[1] - vpy1) <= 30):
            continue
        break

    vpy2 = cp[1] + (cp[1] - vpy1)

    return SVG_CUBIC_FUNCTION_TEMPLATE.format(
        id=id,
        x1=x1, y1=y[0],
        vpx1=vpx1, vpy1=vpy1, vpx2=vpx2,
        cp1=cp[0], cp2=cp[1],
        vpx3=vpx3, vpy2=vpy2, vpx4=vpx4,
        x2=x2, y2=y[1],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width
    )

def draw_cubic_dash_function(id, w, h, min_length, max_stroke_width, rng): # 点線の三次関数
    stroke_color = rng.randint(240, size=3)
    stroke_width = rng.uniform(low=1, high=max_stroke_width+1)

    x1 = rng.randint(0, 50)
    y = rng.randint(100, 300, size=2)
    cp = rng.randint(150, 250, size=2)
    x2 = (2 * cp[0]) - x1

    vpx1 = x1 + ((cp[0] - x1) / 3)
    vpx2 = vpx1 + ((cp[0] - x1) / 3)
    vpx3 = cp[0] + ((cp[0] - x1) / 3)
    vpx4 = vpx3 + ((cp[0] - x1) / 3)

    while True:
        vpy1 = rng.randint(h)
        if (-30 <= (cp[1] - vpy1)) and ((cp[1] - vpy1) <= 30):
            continue
        break

    vpy2 = cp[1] + (cp[1] - vpy1)

    while True:
        sd1 = stroke_width * random.randrange(5, 13, 1) / 2
        sd2 = stroke_width * random.randrange(2, 10, 1) / 2
        if sd1 < sd2:
            continue
        break

    return SVG_CUBIC_FUNCTION_dash_TEMPLATE.format(
        id=id,
        x1=x1, y1=y[0],
        vpx1=vpx1, vpy1=vpy1, vpx2=vpx2,
        cp1=cp[0], cp2=cp[1],
        vpx3=vpx3, vpy2=vpy2, vpx4=vpx4,
        x2=x2, y2=y[1],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=sd1, sd2=sd2
    )

def draw_cubic_dash12_function(id, w, h, min_length, max_stroke_width, rng): # 鎖線の三次関数
    stroke_color = rng.randint(240, size=3)
    stroke_width = rng.uniform(low=1, high=max_stroke_width+1)

    x1 = rng.randint(0, 50)
    y = rng.randint(100, 300, size=2)
    cp = rng.randint(150, 250, size=2)
    x2 = (2 * cp[0]) - x1

    vpx1 = x1 + ((cp[0] - x1) / 3)
    vpx2 = vpx1 + ((cp[0] - x1) / 3)
    vpx3 = cp[0] + ((cp[0] - x1) / 3)
    vpx4 = vpx3 + ((cp[0] - x1) / 3)

    while True:
        vpy1 = rng.randint(h)
        if (-30 <= (cp[1] - vpy1)) and ((cp[1] - vpy1) <= 30):
            continue
        break

    vpy2 = cp[1] + (cp[1] - vpy1)

    sd1 = stroke_width * random.randrange(5, 13, 1) / 2
    sde = stroke_width * random.randrange(3, 5, 1) / 2
    sdf = stroke_width * random.randrange(2, 4, 1) / 2

    dc = rng.randint(0, 2)

    if dc == 0:
        return SVG_CUBIC_FUNCTION_dash1_TEMPLATE.format(
            id=id,
            x1=x1, y1=y[0],
            vpx1=vpx1, vpy1=vpy1, vpx2=vpx2,
            cp1=cp[0], cp2=cp[1],
            vpx3=vpx3, vpy2=vpy2, vpx4=vpx4,
            x2=x2, y2=y[1],
            r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
            sw=stroke_width,
            sd1=sd1, sde=sde, sdf=sdf
        )
    else:
        return SVG_CUBIC_FUNCTION_dash2_TEMPLATE.format(
            id=id,
            x1=x1, y1=y[0],
            vpx1=vpx1, vpy1=vpy1, vpx2=vpx2,
            cp1=cp[0], cp2=cp[1],
            vpx3=vpx3, vpy2=vpy2, vpx4=vpx4,
            x2=x2, y2=y[1],
            r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
            sw=stroke_width,
            sd1=sd1, sde=sde, sdf=sdf
        )

"""def draw_circle(id, w, h, min_length, max_stroke_width, rng): # 円
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = max_stroke_width
    #circle_circle
    x = rng.randint(w, size=1)
    y = rng.randint(h, size=1)
    r = rng.randint(1, w/2, size=1)

    return SVG_CIRCLE_TEMPLATE.format(
        id=id,
        cx=x[0], cy=y[0],
        ra=r[0],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width
    )

    #path_circle
    #x = rng.randint(w, size=2)
    #y = rng.randint(h, size=1)
    #r1 = rng.randint(w/16, size=1)
    #r2 = rng.randint(w/16, size=1)

    #return SVG_CIRCLE_TEMPLATE.format(
        #id=id,
        #x1=x[0], x2=x[1],
        #y1=y[0], y2=y[0],
        #rx=r1[0], ry=r2[0],
        #r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        #sw=stroke_width
    #)

def draw_circle_dash(id, w, h, min_length, max_stroke_width, rng):
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = max_stroke_width
    # circle_circle
    x = rng.randint(w, size=1)
    y = rng.randint(h, size=1)
    r = rng.randint(1, w/2, size=1)

    return SVG_CIRCLE_dash_TEMPLATE.format(
        id=id,
        cx=x[0], cy=y[0],
        ra=r[0],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=random.randrange(6, 11, 1),
        sd2=random.randrange(4, 8, 1)
    )

    #path_circle_dash
    #x = rng.randint(w, size=2)
    #y = rng.randint(h, size=1)
    #r1 = rng.randint(w/16, size=1)
    #r2 = rng.randint(w/16, size=1)

    #return SVG_CIRCLE_dash_TEMPLATE.format(
        #id=id,
        #x1=x[0], x2=x[1],
        #y1=y[0], y2=y[0],
        #rx=r1[0], ry=r2[0],
        #r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        #sw=stroke_width,
        #sd1=random.randrange(6, 11, 1),
        #sd2=random.randrange(4, 8, 1)
    #)"""

"""def draw_rect(id, w, h, min_length, max_stroke_width, rng): # 長方形
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = max_stroke_width
    x = rng.randint(w-5, size=1)
    y = rng.randint(h-5, size=1)
    w = rng.randint(5, w-x, size=1)
    h = rng.randint(5, h-y, size=1)

    return SVG_RECT_TEMPLATE.format(
        id=id,
        x=x[0], y=y[0],
        w=w[0], h=h[0],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width
    )

def draw_rect_dash(id, w, h, min_length, max_stroke_width, rng):
    stroke_color = rng.randint(240, size=3)
    # stroke_width = rng.randint(low=1, high=max_stroke_width+1)
    stroke_width = max_stroke_width
    x = rng.randint(w-5, size=1)
    y = rng.randint(h-5, size=1)
    w = rng.randint(5, w-x, size=1)
    h = rng.randint(5, h-y, size=1)

    return SVG_RECT_dash_TEMPLATE.format(
        id=id,
        x=x[0], y=y[0],
        w=w[0], h=h[0],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2],
        sw=stroke_width,
        sd1=random.randrange(6, 11, 1),
        sd2=random.randrange(4, 8, 1)
    )"""

# 描かれる線種
def draw_path(stroke_type, id, w, h, min_length, max_stroke_width, rng):
    if stroke_type == 2:
        stroke_type = rng.randint(24) # 使用する関数の数だけランダムの幅を増やす
    path_selector = {
        0: draw_line,
        1: draw_cubic_bezier_curve,
        2: draw_line_dash,
        3: draw_cubic_bezier_curve_dash,
        4: draw_line_dash1,
        5: draw_line_dash2,
        6: draw_cubic_bezier_curve_dash1,
        7: draw_cubic_bezier_curve_dash2,
        8: draw_line,
        9: draw_cubic_bezier_curve,
        10: draw_line_dash,
        11: draw_cubic_bezier_curve_dash,
        # 12: draw_line,
        # 13: draw_cubic_bezier_curve,
        # 14: draw_line,
        # 15: draw_cubic_bezier_curve,
        12: draw_quadratic_function,
        13: draw_quadratic_dash_function,
        14: draw_quadratic_dash12_function,
        15: draw_cubic_function,
        16: draw_cubic_dash_function,
        17: draw_cubic_dash12_function,
        18: draw_quadratic_function,
        19: draw_quadratic_dash_function,
        20: draw_quadratic_dash12_function,
        21: draw_cubic_function,
        22: draw_cubic_dash_function,
        23: draw_cubic_dash12_function
        # 2: draw_line_contact,
        # 3: draw_quadratic_contact,
        # 6: draw_rect,
        # 7: draw_rect_dash
    }

    return path_selector[stroke_type](id, w, h, min_length, max_stroke_width, rng)


def gen_data(data_dir, config, rng, num_train, num_test):
    file_list = []
    num = num_train + num_test
    for file_id in range(num):
        while True:
            svg = SVG_START_TEMPLATE.format(
                w=config.width,
                h=config.height,
            )
            svgpre = SVG_START_TEMPLATE
            # co = rng.randint(22, 34, 2) # 接点、交点を決める
            for i in range(rng.randint(2, config.num_strokes)):
                path = draw_path(
                    stroke_type=config.stroke_type,
                    id=i,
                    w=config.width,
                    h=config.height,
                    min_length=config.min_length,
                    max_stroke_width=config.max_stroke_width,
                    rng=rng,
                )
                svg += path + '\n'
                svgpre += path + '\n'

            svg += SVG_END_TEMPLATE
            svgpre += SVG_END_TEMPLATE
            s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
            s_img = Image.open(io.BytesIO(s_png))
            s = np.array(s_img)[:, :, 3].astype(np.float)  # / 255.0
            max_intensity = np.amax(s)

            if max_intensity == 0:
                continue
            else:
                s = s / max_intensity  # [0,1]
            break

        if file_id < num_train:
            cat = 'train'
        else:
            cat = 'test'

        # svgpre
        svgpre_file_path = os.path.join(data_dir, cat, '%d.svg_pre' % file_id)
        print(svgpre_file_path)
        with open(svgpre_file_path, 'w') as f:
            f.write(svgpre)

        # svg and jpg for reference
        svg_dir = os.path.join(data_dir, 'svg')
        if not os.path.exists(svg_dir):
            os.makedirs(svg_dir)
        jpg_dir = os.path.join(data_dir, 'jpg')
        if not os.path.exists(jpg_dir):
            os.makedirs(jpg_dir)

        svg_file_path = os.path.join(data_dir, 'svg', '%d.svg' % file_id)
        jpg_file_path = os.path.join(data_dir, 'jpg', '%d.jpg' % file_id)

        with open(svg_file_path, 'w') as f:
            f.write(svg)
        s_img.convert('RGB').save(jpg_file_path)

        if file_id < num_train:
            file_list.append(svgpre_file_path)

    return file_list

# 本研究では使用しない
def preprocess_path(file_path, w, h, rng, is_binary):
# def preprocess_path(file_path, w, h, rng):
    with open(file_path, 'r') as f:
        svg = f.read()

    svg = svg.format(w=w, h=h)
    img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(img))
    s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
    max_intensity = np.amax(s)
    if is_binary:
        s = (s > 0) * max_intensity  # 2値化 # data_line_2
    s = s / max_intensity

    # while True:
    svg_xml = et.fromstring(svg)
    path_id = rng.randint(len(svg_xml[0]))
    # path_id = rng.randint(1, len(svg_xml[0])) # data_line_2
    svg_xml[0][0] = svg_xml[0][path_id]
    del svg_xml[0][1:]  # [0][2:] # data_line_2
    svg_one = et.tostring(svg_xml, method='xml')

    # leave only one path
    y_png = cairosvg.svg2png(bytestring=svg_one)
    y_img = Image.open(io.BytesIO(y_png))
    # y = np.array(y_img)[:, :, 3].astype(np.float) / max_intensity  # [0,1] # data_line ori
    y = np.array(y_img)[:, :, 3].astype(np.float)  # [0,1] # data_line_2 st
    if is_binary:
        y = (y > 0) * max_intensity  # 2値化
    y = y / max_intensity # data_line_2 end

    pixel_ids = np.nonzero(y)
    # if len(pixel_ids[0]) == 0:
    #     continue
    # else:
    #     break            

    # select arbitrary marking pixel
    point_id = rng.randint(len(pixel_ids[0]))
    px, py = pixel_ids[0][point_id], pixel_ids[1][point_id]

    y = np.reshape(y, [h, w, 1])
    x = np.zeros([h, w, 2])
    x[:, :, 0] = s
    x[px, py, 1] = 1.0

    # # debug
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.subplot(222)
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.subplot(223)
    # plt.imshow(np.concatenate((x, np.zeros([h, w, 1])), axis=-1))
    # plt.subplot(224)
    # plt.imshow(y[:,:,0], cmap=plt.cm.gray)
    # plt.show()

    return x, y

# 学習に使用する
def preprocess_overlap(file_path, w, h, rng, is_binary):
    with open(file_path, 'r') as f:
        svg = f.read()
    #svg = svg.format(w=w, h=h)
    svg_xml = et.fromstring(svg)
    svg_xml.attrib["height"] = "512"
    svg_xml.attrib["width"] = "512"
    svg = et.tostring(svg_xml, method="xml")
    img = cairosvg.svg2png(bytestring=svg)
    #img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(img))
    s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
    max_intensity = np.amax(s)
    if is_binary:
       s = (s > 0) * max_intensity  # 2値化
    s = s / max_intensity

    # while True:
    path_list = []  # data_line ori
    # svg_xml = et.fromstring(svg)
    num_paths = len(svg_xml[0])  # tikz_graph → svg_xml[1]
    tikz_svg = 0
    if num_paths == 1:
        num_paths = len(svg_xml[1])
        tikz_svg = 1

    for i in range(0, num_paths):  # data_line_2 st
        if tikz_svg == 0:
            if "stroke-dasharray" in svg_xml[0][i].attrib.keys():  # 破線を実線に変える
                del svg_xml[0][i].attrib["stroke-dasharray"]
        else:
            if "stroke-dasharray" in svg_xml[1][i].attrib.keys():  # 破線を実線に変える
                del svg_xml[1][i].attrib["stroke-dasharray"]
    svg_str = et.tostring(svg_xml, method='xml')

    # leave only one path
    y_png = cairosvg.svg2png(bytestring=svg_str)
    y_img = Image.open(io.BytesIO(y_png))
    # y = np.array(y_img)[:, :, 3].astype(np.float) / max_intensity
    # y = np.reshape(y, [h, w, 1])
    y = np.array(y_img)[:, :, 3].astype(np.float)  # [0,1]
    if is_binary:
        y = (y > 0) * max_intensity  # 2値化
    y = y / max_intensity
    path = (np.array(y_img)[:, :, 3] > 0)

    x = np.expand_dims(s, axis=-1)
    y = np.reshape(y, [h, w, 1]) # 実線化のための正解y
    y = np.expand_dims(y, axis=-1) # data_line ori
    # y = np.expand_dims(path, axis=-1) # data_line_2

    # # debug
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(img)
    # plt.subplot(132)
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.subplot(133)
    # plt.imshow(y[:,:,0], cmap=plt.cm.gray)
    # plt.show()

    return x, y


def main(config):
    prepare_dirs_and_logger(config)
    batch_manager = BatchManager(config)
    # preprocess_path('data/line/train/0.svg_pre', 64, 64, batch_manager.rng, False)
    preprocess_overlap('data/line/train/0.svg_pre', 512, 512, batch_manager.rng, True)

    # thread test
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = False
    sess = tf.Session(config=sess_config)
    batch_manager.start_thread(sess)

    x, y = batch_manager.batch()
    if config.data_format == 'NCHW':
        x = nhwc_to_nchw(x)
    x_, y_ = sess.run([x, y])
    batch_manager.stop_thread()

    if config.data_format == 'NCHW':
        x_ = x_.transpose([0, 2, 3, 1])

    if config.archi == 'path':
        b_ch = np.zeros([config.batch_size, config.height, config.width, 1])
        x_ = np.concatenate((x_ * 255, b_ch), axis=-1)
    else:
        x_ = x_ * 255
    y_ = y_ * 255

    save_image(x_, '{}/x_fixed.png'.format(config.model_dir))
    save_image(y_, '{}/y_fixed.png'.format(config.model_dir))

    # random pick from parameter space
    x_samples, x_gt, y_gt, sample_list = batch_manager.random_list(8)
    save_image(x_gt, '{}/x_gt.png'.format(config.model_dir))
    save_image(y_gt, '{}/y_gt.png'.format(config.model_dir))

    with open('{}/sample_list.txt'.format(config.model_dir), 'w') as f:
        for sample in sample_list:
            f.write(sample + '\n')

    print('batch manager test done')


if __name__ == "__main__":
    from config import get_config
    from utils import prepare_dirs_and_logger, save_config, save_image

    config, unparsed = get_config()
    setattr(config, 'archi', 'path')  # overlap
    setattr(config, 'dataset', 'line')
    setattr(config, 'width', 512)
    setattr(config, 'height', 512)

    main(config)
