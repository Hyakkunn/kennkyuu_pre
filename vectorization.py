from __future__ import print_function

import warnings
warnings.simplefilter('ignore', FutureWarning)

import os
from tqdm import trange
import multiprocessing
import time
from datetime import datetime
import platform
from subprocess import call
from shutil import copyfile

import numpy as np
import sklearn.neighbors
import skimage.measure

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.misc

from models import *
from utils import save_image
from PIL import Image

from config import get_config
from utils import prepare_dirs_and_logger, save_config

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
import cv2

from ops import *


class BatchManager(object):
    def __init__(self, config):
        self.root = config.data_path
        self.rng = np.random.RandomState(config.random_seed)

        self.test_paths = [r"\mydata\test\{}".format(config.img_path)]

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
        self.is_binary = config.is_binary

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
                x_, y_ = preprocess_path(file_path, self.width, self.height, self.rng, self.is_binary)
            else:
                x_, y_ = preprocess_overlap(file_path, self.width, self.height, self.rng, self.is_binary)
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
                x, y = preprocess_path(file_path, self.width, self.height, self.rng, self.is_binary)
            else:
                x, y = preprocess_overlap(file_path, self.width, self.height, self.rng, self.is_binary)
            x_list.append(x)

            if self.is_pathnet:
                b_ch = np.zeros([self.height, self.width, 1])
                xs.append(np.concatenate((x * 255, b_ch), axis=-1))
            else:
                xs.append(x * 255)
            ys.append(y * 255)

        return np.array(x_list), np.array(xs), np.array(ys), file_list

    def read_svg(self, file_path):
        with open(file_path, 'r') as f:
            svg = f.read()

        svg = svg.format(w=self.width, h=self.height)
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
        if (file_path == r'data\line/test\47648.svg_pre'):
            del s
            img_test = Image.open(r'mydata/test\2axis002.png').convert('L')
            s = 255 - np.array(img_test)
            # import matplotlib.pyplot as plt
            # plt.imshow(s)
            # plt.show()
            print(np.unique(s)[-2])
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > 0) * max_intensity  # 2値化
        s = s / max_intensity
        # np.save('svg_array', s)

        path_list = []
        svg_xml = et.fromstring(svg)
        num_paths = len(svg_xml[0])

        for i in range(1, num_paths):
            svg_xml = et.fromstring(svg)
            svg_xml[0][1] = svg_xml[0][i]
            del svg_xml[0][2:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            path = (np.array(y_img)[:, :, 3] > 0)
            if self.is_binary:
                path = (path > 0) * max_intensity  # 二値化
            path_list.append(path)

        return s, num_paths, path_list

    def read_bmp(self, file_path):
        img_test = Image.open(file_path).convert('L')
        s = 255 - np.array(img_test)
        # import matplotlib.pyplot as plt
        # plt.imshow(s)
        # plt.show()
        print(np.unique(s)[-2])
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > 0) * max_intensity  # 2値化
        s = s / max_intensity
        # np.save('svg_array', s)

        return s


def preprocess_path(file_path, w, h, rng, is_binary):
    with open(file_path, 'r') as f:
        svg = f.read()

    svg = svg.format(w=w, h=h)
    img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(img))
    s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
    # sigmaX = rng.randint(10)
    # s = cv2.GaussianBlur(s, ksize=(3, 3), sigmaX=sigmaX)
    max_intensity = np.amax(s)
    if is_binary:
        s = (s > 0) * max_intensity  # 2値化
    s = s / max_intensity

    # while True:
    svg_xml = et.fromstring(svg)
    path_id = rng.randint(1, len(svg_xml[0]))
    svg_xml[0][1] = svg_xml[0][path_id]
    del svg_xml[0][2:]
    svg_one = et.tostring(svg_xml, method='xml')

    # leave only one path
    y_png = cairosvg.svg2png(bytestring=svg_one)
    y_img = Image.open(io.BytesIO(y_png))
    y = np.array(y_img)[:, :, 3].astype(np.float)  # [0,1]
    if is_binary:
        y = (y > 0) * max_intensity  # 2値化
    y = y / max_intensity

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


def preprocess_overlap(file_path, w, h, rng, is_binary):
    with open(file_path, 'r') as f:
        svg = f.read()
    svg = svg.format(w=w, h=h)
    img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(img))
    s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
    # sigmaX = rng.randint(10)
    # s = cv2.GaussianBlur(s, ksize=(3, 3), sigmaX=sigmaX)
    max_intensity = np.amax(s)
    if is_binary:
        s = (s > 0) * max_intensity  # 2値化
    s = s / max_intensity

    # while True:
    path_list = []
    svg_xml = et.fromstring(svg)
    num_paths = len(svg_xml[0])

    for i in range(1, num_paths):
        svg_xml[0][i].attrib['stroke-width'] = str(0.1)
    svg_str = et.tostring(svg_xml, method='xml')

    # leave only one path
    y_png = cairosvg.svg2png(bytestring=svg_str)
    y_img = Image.open(io.BytesIO(y_png))
    path = (np.array(y_img)[:, :, 3] > 0)

    x = np.expand_dims(s, axis=-1)
    y = np.expand_dims(path, axis=-1)

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


# def preprocess_overlap(file_path, w, h, rng, is_binary):
#     with open(file_path, 'r') as f:
#         svg = f.read()
#     svg = svg.format(w=w, h=h)
#     img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
#     img = Image.open(io.BytesIO(img))
#     s = np.array(img)[:,:,3].astype(np.float) # / 255.0
#     # sigmaX = rng.randint(10)
#     # s = cv2.GaussianBlur(s, ksize=(3, 3), sigmaX=sigmaX)
#     max_intensity = np.amax(s)
#     if is_binary:
#         s = (s > 0) * max_intensity  # 2値化
#     s = s / max_intensity
#
#     # while True:
#     path_list = []
#     svg_xml = et.fromstring(svg)
#     num_paths = len(svg_xml[0])
#
#     for i in range(1, num_paths):
#         svg_xml = et.fromstring(svg)
#         svg_xml[0][1] = svg_xml[0][i]
#         del svg_xml[0][2:]
#         svg_one = et.tostring(svg_xml, method='xml')
#
#         # leave only one path
#         y_png = cairosvg.svg2png(bytestring=svg_one)
#         y_img = Image.open(io.BytesIO(y_png))
#         path = (np.array(y_img)[:,:,3] > 0)
#         path_list.append(path)
#
#     y = np.zeros([h, w], dtype=np.int)
#     for i in range(num_paths-2):
#         for j in range(i+1, num_paths-1):
#             intersect = np.logical_and(path_list[i], path_list[j])
#             y = np.logical_or(intersect, y)
#
#     x = np.expand_dims(s, axis=-1)
#     y = np.expand_dims(y, axis=-1)
#
#     # # debug
#     # plt.figure()
#     # plt.subplot(131)
#     # plt.imshow(img)
#     # plt.subplot(132)
#     # plt.imshow(s, cmap=plt.cm.gray)
#     # plt.subplot(133)
#     # plt.imshow(y[:,:,0], cmap=plt.cm.gray)
#     # plt.show()
#
#     return x, y

# ---------------------------------------------------------
# testコード
# ----------------------------------------------------------
class Param(object):
    pass


def vectorize_mp(q):
    while True:
        pm = q.get()
        if pm is None:
            break

        vectorize(pm)
        print('%s: qsize %d' % (datetime.now(), q.qsize()))
        q.task_done()


class Tester(object):
    def __init__(self, config, batch_manager):
        tf.set_random_seed(config.random_seed)
        self.config = config
        self.batch_manager = batch_manager
        self.rng = self.batch_manager.rng

        self.b_num = config.test_batch_size
        self.height = config.height
        self.width = config.width
        self.conv_hidden_num = config.conv_hidden_num
        self.repeat_num = config.repeat_num
        self.data_format = config.data_format
        self.use_norm = config.use_norm

        self.load_pathnet = config.load_pathnet
        self.load_overlapnet = config.load_overlapnet
        self.find_overlap = config.find_overlap
        self.overlap_threshold = config.overlap_threshold
        self.max_label = config.max_label
        self.label_cost = config.label_cost
        self.sigma_neighbor = config.sigma_neighbor
        self.sigma_predict = config.sigma_predict
        self.neighbor_sample = config.neighbor_sample

        self.num_test = config.num_test
        self.test_paths = self.batch_manager.test_paths
        if config.dataset == 'baseball' or config.dataset == 'cat' or \
                config.dataset == 'multi':
            self.test_paths = self.batch_manager.vec_paths
        if self.num_test < len(self.test_paths):
            self.test_paths = self.rng.choice(self.test_paths, self.num_test, replace=False)
            self.test_paths = np.insert(self.test_paths, 0, r'data\line/test\47648.svg_pre')
        self.mp = config.mp
        self.num_worker = config.num_worker

        self.model_dir = config.model_dir
        self.data_path = config.data_path
        self.model = config.model

        self.build_model()

    def build_model(self):
        pathnet_graph = tf.Graph()
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=tf.GPUOptions(allow_growth=True))
        self.sp = tf.Session(config=sess_config, graph=pathnet_graph)
        with pathnet_graph.as_default():
            self.xp = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 2])
            if self.data_format == 'NCHW':
                self.xp = nhwc_to_nchw(self.xp)

            if self.model == 'VDSR':
                self.yp, _ = VDSR(self.xp, self.conv_hidden_num, self.repeat_num,
                                  self.data_format, self.use_norm, train=False)
            if self.model == 'UNet':
                self.yp, _ = UNet(self.xp, self.conv_hidden_num, self.repeat_num,
                                  self.data_format, self.use_norm, train=False)
            show_all_variables()

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.load_pathnet)
            assert (ckpt and self.load_pathnet)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sp, os.path.join(self.load_pathnet, ckpt_name))
            print('%s: Pre-trained model restored from %s' % (datetime.now(), self.load_pathnet))

        if self.find_overlap:
            overlapnet_graph = tf.Graph()
            self.so = tf.Session(config=sess_config, graph=overlapnet_graph)
            with overlapnet_graph.as_default():
                self.xo = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 1])
                if self.data_format == 'NCHW':
                    self.xo = nhwc_to_nchw(self.xo)

                if self.model == 'VDSR':
                    self.yo, _ = VDSR(self.xo, self.conv_hidden_num, self.repeat_num,
                                      self.data_format, self.use_norm, train=False)

                if self.model == 'UNet':
                    self.yo, _ = UNet(self.xo, self.conv_hidden_num, self.repeat_num,
                                      self.data_format, self.use_norm, train=False)

                show_all_variables()

                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(self.load_overlapnet)
                assert (ckpt and self.load_overlapnet)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(self.so, os.path.join(self.load_overlapnet, ckpt_name))
                print('%s: Pre-trained model restored from %s' % (datetime.now(), self.load_overlapnet))

    def t_test(self):
        if self.mp:
            q = multiprocessing.JoinableQueue()
            pool = multiprocessing.Pool(self.num_worker, vectorize_mp, (q,))

        # preprocess first
        for i in trange(self.num_test):
            file_path = self.test_paths[i]
            print('\n[{}/{}] start prediction, path: {}'.format(i + 1, self.num_test, file_path))

            param = self.predict(file_path)

            if self.mp:
                q.put(param)
            else:
                vectorize(param)

        if self.mp:
            q.join()
            pool.terminate()
            pool.join()

        self.stat()

    def predict(self, file_path):
        # convert svg to raster image
        # img, num_paths, path_list = self.batch_manager.read_svg(file_path)
        img = self.batch_manager.read_bmp(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        input_img_path = os.path.join(self.model_dir, '%s_0_input.png' % file_name)
        save_image((1 - img[np.newaxis, :, :, np.newaxis]) * 255, input_img_path, padding=0)

        # # debug
        # print(num_paths)
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.show()

        pm = Param()

        # predict paths through pathnet
        start_time = time.time()
        paths, path_pixels = self.extract_path(img)
        num_path_pixels = len(path_pixels[0])
        pids = self.rng.randint(num_path_pixels, size=8)
        path_img_path = os.path.join(self.model_dir, '%s_1_path.png' % file_name)
        save_image((1 - paths[pids, :, :, :]) * 255, path_img_path, padding=0)

        # # debug
        # plt.imshow(paths[0,:,:,0], cmap=plt.cm.gray)
        # plt.show()

        duration = time.time() - start_time
        print('%s: %s, predict paths (#pixels:%d) through pathnet (%.3f sec)' % (
            datetime.now(), file_name, num_path_pixels, duration))
        pm.duration_pred = duration
        pm.duration = duration

        dup_dict = {}
        dup_rev_dict = {}
        dup_id = num_path_pixels  # start id of duplicated pixels

        if self.find_overlap:
            # predict overlap using overlap net
            start_time = time.time()
            ov = self.overlap(img)

            overlap_img_path = os.path.join(self.model_dir, '%s_2_overlap.png' % file_name)
            ov_img = ov[np.newaxis, :, :, np.newaxis]
            save_image((1 - ov_img) * 255, overlap_img_path, padding=0)

            # # debug
            # plt.imshow(ov, cmap=plt.cm.gray)
            # plt.show()

            for i in range(num_path_pixels):
                if ov[path_pixels[0][i], path_pixels[1][i]]:
                    dup_dict[i] = dup_id
                    dup_rev_dict[dup_id] = i
                    dup_id += 1

            # debug
            # print(dup_dict)
            # print(dup_rev_dict)

            duration = time.time() - start_time
            print('%s: %s, predict overlap (#:%d) through ovnet (%.3f sec)' % (
                datetime.now(), file_name, dup_id - num_path_pixels, duration))
            pm.duration_ov = duration
            pm.duration += duration
        else:
            pm.duration_ov = 0

        # write config file for graphcut
        start_time = time.time()
        tmp_dir = os.path.join(self.model_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        pred_file_path = os.path.join(tmp_dir, file_name + '.pred')
        f = open(pred_file_path, 'w')
        # info
        f.write(pred_file_path + '\n')
        f.write(self.data_path + '\n')
        f.write('%d\n' % self.max_label)
        f.write('%d\n' % self.label_cost)
        f.write('%f\n' % self.sigma_neighbor)
        f.write('%f\n' % self.sigma_predict)
        # f.write('%d\n' % num_path_pixels)
        f.write('%d\n' % dup_id)

        # support only symmetric edge weight
        radius = self.sigma_neighbor * 2
        nb = sklearn.neighbors.NearestNeighbors(radius=radius)
        nb.fit(np.array(path_pixels).transpose())

        high_spatial = 100000
        for i in range(num_path_pixels - 1):
            p1 = np.array([path_pixels[0][i], path_pixels[1][i]])
            pred_p1 = np.reshape(paths[i, :, :, :], [self.height, self.width])

            # see close neighbors and some far neighbors (stochastic sampling)
            rng = nb.radius_neighbors([p1])
            num_close = len(rng[1][0])
            far = np.setdiff1d(range(i + 1, num_path_pixels), rng[1][0])
            num_far = len(far)
            num_far = int(num_far * self.neighbor_sample)
            if num_far > 0:
                far_ids = self.rng.choice(far, size=num_far)
                nb_ids = np.concatenate((rng[1][0], far_ids))
            else:
                nb_ids = rng[1][0]

            for rj, j in enumerate(nb_ids):  # ids
                if j <= i:
                    continue
                p2 = np.array([path_pixels[0][j], path_pixels[1][j]])
                if rj < num_close:
                    d12 = rng[0][0][rj]
                else:
                    d12 = np.linalg.norm(p1 - p2, 2)

                # for j in range(i+1, num_path_pixels): # see entire neighbors
                #     p2 = np.array([path_pixels[0][j], path_pixels[1][j]])
                #     d12 = np.linalg.norm(p1-p2, 2)

                pred_p2 = np.reshape(paths[j, :, :, :], [self.height, self.width])
                pred = (pred_p1[p2[0], p2[1]] + pred_p2[p1[0], p1[1]]) * 0.5
                pred = np.exp(-0.5 * (1.0 - pred) ** 2 / self.sigma_predict ** 2)

                spatial = np.exp(-0.5 * d12 ** 2 / self.sigma_neighbor ** 2)
                f.write('%d %d %f %f\n' % (i, j, pred, spatial))

                dup_i = dup_dict.get(i)
                if dup_i is not None:
                    f.write('%d %d %f %f\n' % (j, dup_i, pred, spatial))  # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (i, dup_i, 0, high_spatial))  # shouldn't be labeled together
                dup_j = dup_dict.get(j)
                if dup_j is not None:
                    f.write('%d %d %f %f\n' % (i, dup_j, pred, spatial))  # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (j, dup_j, 0, high_spatial))  # shouldn't be labeled together

                if dup_i is not None and dup_j is not None:
                    f.write('%d %d %f %f\n' % (dup_i, dup_j, pred, spatial))  # dup_i < dup_j

        f.close()
        duration = time.time() - start_time
        print('%s: %s, prediction computed (%.3f sec)' % (datetime.now(), file_name, duration))
        pm.duration_map = duration
        pm.duration += duration

        pm.path_pixels = path_pixels
        pm.dup_dict = dup_dict
        pm.dup_rev_dict = dup_rev_dict
        pm.img = img
        pm.file_path = file_path
        pm.model_dir = self.model_dir
        pm.height = self.height
        pm.width = self.width
        pm.max_label = self.max_label
        pm.sigma_neighbor = self.sigma_neighbor
        pm.sigma_predict = self.sigma_predict

        return pm

    def extract_path(self, img):
        path_pixels = np.nonzero(img)
        num_path_pixels = len(path_pixels[0])
        assert (num_path_pixels > 0)

        y_batch = None
        for b in range(0, num_path_pixels, self.b_num):
            b_size = min(self.b_num, num_path_pixels - b)
            x_batch = np.zeros([b_size, self.height, self.width, 2])
            for i in range(b_size):
                x_batch[i, :, :, 0] = img
                px, py = path_pixels[0][b + i], path_pixels[1][b + i]
                x_batch[i, px, py, 1] = 1.0

            if self.data_format == 'NCHW':
                x_batch = to_nchw_numpy(x_batch)
            y_b = self.sp.run(self.yp, feed_dict={self.xp: x_batch})
            y_b = np.clip(y_b, 0, 1)
            if self.data_format == 'NCHW':
                y_b = to_nhwc_numpy(y_b)
            if y_batch is None:
                y_batch = y_b
            else:
                y_batch = np.concatenate((y_batch, y_b), axis=0)

        return y_batch, path_pixels

    def overlap(self, img):
        x_batch = np.zeros([1, self.height, self.width, 1])
        x_batch[0, :, :, 0] = img

        if self.data_format == 'NCHW':
            x_batch = to_nchw_numpy(x_batch)
        y_b = self.so.run(self.yo, feed_dict={self.xo: x_batch})
        if self.data_format == 'NCHW':
            y_b = to_nhwc_numpy(y_b)
        return (y_b[0, :, :, 0] >= self.overlap_threshold)

    def stat(self):
        from glob import glob
        stat_paths = sorted(glob("{}/*{}".format(self.model_dir, '_stat.txt')))
        diff = []
        abs_diff = []
        acc = []
        d_pred = []
        d_ov = []
        d_map = []
        d_vec = []
        duration = []

        # print(len(stat_paths))
        for path in stat_paths:
            with open(path, 'r') as f:
                stat = f.readline()
                # print(stat)
            stat = stat.split()
            # file_path, num_labels, pm.num_paths, acc_avg,
            # duration_pred, duration_ov, duration_map,
            # duration_vect, duration
            num_labels = int(stat[1])
            gt_labels = int(stat[2])
            acc_ = float(stat[3])
            dpred = float(stat[4])
            dov = float(stat[5])
            dmap = float(stat[6])
            dvec = float(stat[7])
            d = float(stat[8])

            diff.append(num_labels - gt_labels)
            abs_diff.append(abs(num_labels - gt_labels))
            acc.append(acc_)
            d_pred.append(dpred)
            d_ov.append(dov)
            d_map.append(dmap)
            d_vec.append(dvec)
            duration.append(d)

        print('label abs diff: {}'.format(np.average(abs_diff)))
        print('acc: {}'.format(np.average(acc)))
        print('duration for prediction: {}'.format(np.average(d_pred)))
        print('duration for overlap: {}'.format(np.average(d_ov)))
        print('duration for mapping: {}'.format(np.average(d_map)))
        print('duration for vectorization: {}'.format(np.average(d_vec)))
        print('duration total: {}'.format(np.average(duration)))

        stat_path = os.path.join(self.model_dir, 'summary.txt')
        with open(stat_path, 'w') as f:
            f.write('label abs diff: {}\n'.format(np.average(abs_diff)))
            f.write('acc: {}\n'.format(np.average(acc)))
            f.write('duration for prediction: {}\n'.format(np.average(d_pred)))
            f.write('duration for overlap: {}\n'.format(np.average(d_ov)))
            f.write('duration for mapping: {}\n'.format(np.average(d_map)))
            f.write('duration for vectorization: {}\n'.format(np.average(d_vec)))
            f.write('duration total: {}\n'.format(np.average(duration)))


def vectorize(pm):
    start_time = time.time()
    file_path = os.path.basename(pm.file_path)
    file_name = os.path.splitext(file_path)[0]

    # 1. label
    labels, e_before, e_after = label(file_name, pm)

    # 2. merge small components
    labels = merge_small_component(labels, pm)

    # # 2-2. assign one label per one connected component
    # labels = label_cc(labels, pm)

    # 3. compute accuracy
    # accuracy_list = compute_accuracy(labels, pm)

    unique_labels = np.unique(labels)
    num_labels = unique_labels.size
    # acc_avg = np.average(accuracy_list)
    # acc_avg = 0

    # print('%s: %s, the number of labels %d, truth %d' % (datetime.now(), file_name, num_labels, pm.num_paths))
    print('%s: %s, energy before optimization %.4f' % (datetime.now(), file_name, e_before))
    print('%s: %s, energy after optimization %.4f' % (datetime.now(), file_name, e_after))
    # print('%s: %s, accuracy computed, avg.: %.3f' % (datetime.now(), file_name, acc_avg))

    # 4. save image
    save_label_img(labels, unique_labels, num_labels, pm)
    duration = time.time() - start_time
    pm.duration_vect = duration

    # write result
    pm.duration += duration
    print('%s: %s, done (%.3f sec)' % (datetime.now(), file_name, pm.duration))
    stat_file_path = os.path.join(pm.model_dir, file_name + '_stat.txt')
    # with open(stat_file_path, 'w') as f:
    #     f.write('%s %d %d %.3f %.3f %.3f %.3f %.3f %.3f\n' % (
    #         file_path, num_labels, pm.num_paths, acc_avg,
    #         pm.duration_pred, pm.duration_ov, pm.duration_map,
    #         pm.duration_vect, pm.duration))


def label(file_name, pm):
    start_time = time.time()
    working_path = os.getcwd()
    gco_path = os.path.join(working_path, 'gco/build')
    os.chdir(gco_path)

    pred_file_path = os.path.join(working_path, pm.model_dir, 'tmp', file_name + '.pred')
    sys_name = platform.system()
    if sys_name == 'Windows':
        call(['Release/gco.exe', pred_file_path])
    else:
        call(['./gco', pred_file_path])
    os.chdir(working_path)

    # read graphcut result
    label_file_path = os.path.join(pm.model_dir, 'tmp', file_name + '.label')
    f = open(label_file_path, 'r')
    e_before = float(f.readline())
    e_after = float(f.readline())
    labels = np.fromstring(f.read(), dtype=np.int32, sep=' ')
    f.close()
    duration = time.time() - start_time
    print('%s: %s, labeling finished (%.3f sec)' % (datetime.now(), file_name, duration))

    return labels, e_before, e_after


def merge_small_component(labels, pm):
    knb = sklearn.neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knb.fit(np.array(pm.path_pixels).transpose())

    num_path_pixels = len(pm.path_pixels[0])

    for iter in range(2):
        # # debug
        # print('%d-th iter' % iter)

        unique_label = np.unique(labels)
        for i in unique_label:
            i_label_list = np.nonzero(labels == i)

            # handle duplicated pixels
            for j, i_label in enumerate(i_label_list[0]):
                if i_label >= num_path_pixels:
                    i_label_list[0][j] = pm.dup_rev_dict[i_label]

            # connected component analysis on 'i' label map
            i_label_map = np.zeros([pm.height, pm.width], dtype=np.float)
            i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = 1.0
            cc_map, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)

            # # debug
            # print('%d: # labels %d, # cc %d' % (i, num_i_label_pixels, num_cc))
            # plt.imshow(cc_map, cmap='spectral')
            # plt.show()

            # detect small pixel component
            for j in range(num_cc):
                j_cc_list = np.nonzero(cc_map == (j + 1))
                num_j_cc = len(j_cc_list[0])

                # consider only less than 5 pixels component
                if num_j_cc > 4:
                    continue

                # assign dominant label of neighbors using knn
                for k in range(num_j_cc):
                    p1 = np.array([j_cc_list[0][k], j_cc_list[1][k]])
                    _, indices = knb.kneighbors([p1], n_neighbors=5)
                    max_label_nb = np.argmax(np.bincount(labels[indices][0]))
                    labels[indices[0][0]] = max_label_nb

                    # # debug
                    # print(' (%d,%d) %d -> %d' % (p1[0], p1[1], i, max_label_nb))

                    dup = pm.dup_dict.get(indices[0][0])
                    if dup is not None:
                        labels[dup] = max_label_nb

    return labels


def label_cc(labels, pm):
    unique_label = np.unique(labels)
    num_path_pixels = len(pm.path_pixels[0])

    new_label = pm.max_label
    for i in unique_label:
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        # connected component analysis on 'i' label map
        i_label_map = np.zeros([pm.height, pm.width], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = 1.0
        cc_map, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)

        if num_cc > 1:
            for i_label in i_label_list[0]:
                cc_label = cc_map[pm.path_pixels[0][i_label], pm.path_pixels[1][i_label]]
                if cc_label > 1:
                    labels[i_label] = new_label + (cc_label - 2)

            new_label += (num_cc - 1)

    return labels


def compute_accuracy(labels, pm):
    unique_labels = np.unique(labels)
    num_path_pixels = len(pm.path_pixels[0])

    acc_id_list = []
    acc_list = []
    for i in unique_labels:
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        i_label_map = np.zeros([pm.height, pm.width], dtype=np.bool)
        i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = True

        accuracy_list = []
        for j, stroke in enumerate(pm.path_list):
            intersect = np.sum(np.logical_and(i_label_map, stroke))
            union = np.sum(np.logical_or(i_label_map, stroke))
            accuracy = intersect / float(union)
            # print('compare with %d-th path, intersect: %d, union :%d, accuracy %.2f' %
            #     (j, intersect, union, accuracy))
            accuracy_list.append(accuracy)

        id = np.argmax(accuracy_list)
        acc = np.amax(accuracy_list)
        # print('%d-th label, match to %d-th path, max: %.2f' % (i, id, acc))
        # consider only large label set
        # if acc > 0.1:
        acc_id_list.append(id)
        acc_list.append(acc)

    # print('avg: %.2f' % np.average(acc_list))
    return acc_list


def save_label_img(labels, unique_labels, num_labels, pm):
    sys_name = platform.system()

    file_path = os.path.basename(pm.file_path)
    file_name = os.path.splitext(file_path)[0]
    num_path_pixels = len(pm.path_pixels[0])
    # gt_labels = pm.num_paths

    cmap = plt.get_cmap('jet')
    cnorm = colors.Normalize(vmin=0, vmax=num_labels - 1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    label_map = np.ones([pm.height, pm.width, 3], dtype=np.float)
    label_map_t = np.ones([pm.height, pm.width, 3], dtype=np.float)
    first_svg = True
    target_svg_path = os.path.join(pm.model_dir, '%s_%d.svg' % (file_name, num_labels))
    for color_id, i in enumerate(unique_labels):
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        color = np.asarray(cscalarmap.to_rgba(color_id))
        label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = color[:3]

        # save i label map
        i_label_map = np.zeros([pm.height, pm.width], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = pm.img[
            pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]]
        _, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)
        i_label_map_path = os.path.join(pm.model_dir, 'tmp', 'i_%s_%d_%d.bmp' % (file_name, i, num_cc))
        scipy.misc.imsave(i_label_map_path, i_label_map)

        i_label_map = np.ones([pm.height, pm.width, 3], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list], pm.path_pixels[1][i_label_list]] = color[:3]
        label_map_t += i_label_map

        # vectorize using potrace
        color *= 255
        color_hex = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))

        if sys_name == 'Windows':
            potrace_path = os.path.join('potrace-1.16.win64', 'potrace.exe')
            call([potrace_path, '-s', '-i', '-C' + color_hex, i_label_map_path])
        else:
            call(['potrace', '-s', '-i', '-C' + color_hex, i_label_map_path])

        i_label_map_svg = os.path.join(pm.model_dir, 'tmp', 'i_%s_%d_%d.svg' % (file_name, i, num_cc))
        if first_svg:
            copyfile(i_label_map_svg, target_svg_path)
            first_svg = False
        else:
            with open(target_svg_path, 'r') as f:
                target_svg = f.read()

            with open(i_label_map_svg, 'r') as f:
                source_svg = f.read()

            path_start = source_svg.find('<g')
            path_end = source_svg.find('</svg>')

            insert_pos = target_svg.find('</svg>')
            target_svg = target_svg[:insert_pos] + source_svg[path_start:path_end] + target_svg[insert_pos:]

            with open(target_svg_path, 'w') as f:
                f.write(target_svg)

        # remove i label map
        os.remove(i_label_map_path)
        os.remove(i_label_map_svg)

    # set opacity 0.5 to see overlaps
    with open(target_svg_path, 'r') as f:
        target_svg = f.read()

    insert_pos = target_svg.find('<g')
    target_svg = target_svg[:insert_pos] + '<g fill-opacity="0.5">' + target_svg[insert_pos:]
    insert_pos = target_svg.find('</svg>')
    target_svg = target_svg[:insert_pos] + '</g>' + target_svg[insert_pos:]

    with open(target_svg_path, 'w') as f:
        f.write(target_svg)

    label_map_path = os.path.join(pm.model_dir, '%s_%.2f_%.2f_%d.png' % (
        file_name, pm.sigma_neighbor, pm.sigma_predict, num_labels))
    scipy.misc.imsave(label_map_path, label_map)

    label_map_t /= np.amax(label_map_t)
    label_map_path = os.path.join(pm.model_dir, '%s_%.2f_%.2f_%d_t.png' % (
        file_name, pm.sigma_neighbor, pm.sigma_predict, num_labels))
    scipy.misc.imsave(label_map_path, label_map_t)


def main(config):
    prepare_dirs_and_logger(config)
    setattr(config, "img_path", "image_3_ori.png")
    save_config(config)
    print(config.dataset)

    batch_manager = BatchManager(config)
    tester = Tester(config, batch_manager)
    tester.t_test()

    is_binary = True

    img_test = Image.open(r'image_3_ori.png').convert('L')
    s = 255 - np.array(img_test)
    # import matplotlib.pyplot as plt
    # plt.imshow(s)
    # plt.show()
    print(np.unique(s)[-2])

    max_intensity = np.amax(s)
    if is_binary:
        s = (s > 0) * max_intensity  # 2値化
    s = s / max_intensity


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
