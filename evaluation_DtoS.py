from __future__ import print_function

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
import math
from PIL import Image
import cv2
from skimage.morphology import skeletonize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import cairosvg
import io
import xml.etree.ElementTree as et
from data_manager import *

COLOR_LIST = [[0, 0, 255],
              [0, 255, 0],
              [255, 0, 0],
              [0, 255, 255],
              [255, 255, 0],
              [0, 127, 255],
              [127, 0, 255],
              [127, 255, 0],
              [0, 255, 127],
              [255, 127, 0],
              [255, 0, 127],
              [127, 255, 255],
              [255, 127, 255],
              [255, 255, 127],
              [127, 127, 255],
              [127, 255, 127],
              [127, 127, 255]]

class LabelGenerator:
    def __init__(self):
        self.label = 0

    def get(self):
        self.label += 1
        return self.label

class Evaluater:
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

        self.num_test = config.num_test
        self.test_paths = self.batch_manager.test_paths
        if self.num_test < len(self.test_paths):
            self.test_paths = self.rng.choice(self.test_paths, self.num_test, replace=False)
        self.mp = config.mp
        self.num_worker = config.num_worker

        self.model_dir = config.model_dir
        self.data_path = config.data_path
        self.model = config.model
        self.archi = config.archi

        self.build_model()



    def build_model(self):
        nn_graph = tf.Graph()
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=tf.GPUOptions(allow_growth=True))
        self.s = tf.Session(config=sess_config, graph=nn_graph)
        with nn_graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 1])
            if self.data_format == 'NCHW':
                self.x = nhwc_to_nchw(self.x)

            self.y, _ = VDSR(self.x, self.conv_hidden_num, self.repeat_num,
                                       self.data_format, self.use_norm, train=False)

            show_all_variables()

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.load_overlapnet)
            assert(ckpt and self.load_overlapnet)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.s, os.path.join(self.load_overlapnet, ckpt_name))
            print('%s: Pre-trained model restored from %s' % (datetime.now(), self.load_overlapnet))

    def evaluate(self, height, width, model_log): #評価方法の変更
        self.label = LabelGenerator()
        acc_of_dash_to_solid = 0
        # x = 50 # 配列数
        # c = [0] * x
        # ac = [0] * x
        ct = 0
        no = [0] * 300
        true_zero = 0
        insec_img = 0
        f = open('iou/test_iou.txt', 'x') # optionに'x'を使用しているので，同じ名前のファイルが存在すると停止する # 評価のログを記録するテキストファイルの作成
        f.write('logfile_path : {}\n'.format(model_log))
        count = 0
        for i in trange(self.num_test):
            file_path = self.test_paths[i]
            print('\n[{}/{}] start prediction, path: {}'.format(i+1,self.num_test,file_path))
            result, cf, collect_component, png_component = self.predict(file_path, height, width)
            #正解のカウント
            if collect_component == png_component:
                count += 1
            print(count)

            acc_of_dash_to_solid += result["dash_to_solid"]
            # l = 1.0
            # k = 0
            # while True:
            #     if result["dash_to_solid"] >= l:
            #         c[k] += 1
            #         ac[k] += cf
            #         break
            #     l = l - 0.01
            #     k += 1
            #     continue

            p = 0
            while True:
                if cf <= p:
                    no[p] += 1
                    break
                p += 1
                continue

            if cf <= 1:
                ct += 1

            f.write('\n[{}/{}] start prediction, path: {}'.format(i+1,self.num_test,file_path))
            f.write('\nEvaluation of dash to solid: {}'.format(result["dash_to_solid"]))
            # f.write('\nEvaluation 1.0: {} / 0.99: {} / 0.98: {} / 0.97: {} / 0.96: {} / 0.95: {} / 0.94: {} / 0.93: {} / 0.92: {} 0.91: {}'.format(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]))
            f.write('\nSuccess images: {}\n'.format(ct))
            # print('\nEvaluation 1.0: {} / 0.99: {} / 0.98: {} / 0.97: {} / 0.96: {} / 0.95: {} / 0.94: {} / 0.93: {} / 0.92: {} 0.91: {}'.format(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]))
            # print('\nNot_overlap 0: {} / 1: {} / 2: {} / 3: {} / 4: {} / 5: {} / 6: {} / 7: {} / 8: {} / 9: {} / 10: {}'.format(no[0], no[1], no[2], no[3], no[4], no[5], no[6], no[7], no[8], no[9], no[10]))
            print('\nSuccess images: {}'.format(ct))
            print('\nIntersection Success Image: {}'.format(insec_img))
            print('\njc_true = 0: {}'.format(true_zero))

        acc = count / self.num_test
        print(acc)
        sum_acc = acc_of_dash_to_solid / self.num_test
        # n = 1
        # for j in range(1, x):
        #     if c[j] != 0:
        #         ac[j] = ac[j] / c[j]
        #         ac[j] = math.floor((ac[j] * 10 ** n) / (10 ** n))
        print(
            """
            -------------------------------------
            acc of dash to solid: {}
            -------------------------------------
            """.format(sum_acc)
        )
        f.write('\nacc of dash to solid: {}'.format(sum_acc))
        # f.write('\naverage of cf: 0.99: {} / 0.98: {} / 0.97: {} / 0.96: {} / 0.95: {} / 0.94: {} / 0.93: {} / 0.92: {} / 0.91: {}\n'.format(ac[1], ac[2], ac[3], ac[4], ac[5], ac[6], ac[7], ac[8], ac[9]))
        f.write('\nNot overlap\nno : {}'.format(no))
        f.write('\nIntersection Success Image: {}'.format(insec_img))
        f.write('\njc_true = 0: {}'.format(true_zero))
        f.close()

    def predict(self, file_path, height, width): # モルフォロジー演算を使用する場合は，各行の「モルフォロジー」「モル」をオンにする　
        img, num_paths, gt, img2, gt_2, gt_3 = self.batch_manager.read_svg_for_DtoS(file_path, height, width)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        input_img_path = os.path.join(self.model_dir, '%s_input.png' % file_name)
        save_image((1-img[np.newaxis,:,:,np.newaxis])*255, input_img_path, padding=0)
        input_img_path2 = os.path.join(self.model_dir, '%s_input_dash.png' % file_name)
        save_image((1 - img2[np.newaxis, :, :, np.newaxis]) * 255, input_img_path2, padding=0)
        true_img_sk_path = os.path.join(self.model_dir, '%s_true.png' % file_name)
        save_image((1 - gt_2[np.newaxis, :, :, np.newaxis]) * 255, true_img_sk_path, padding=0) #正解画像gt_2
        true_img_sk_path_2 = os.path.join(self.model_dir, '%s_true_sk_1.png' % file_name)
        save_image((1 - gt[np.newaxis, :, :, np.newaxis]) * 255, true_img_sk_path_2, padding=0)
        true_img_sk_path_3 = os.path.join(self.model_dir, '%s_true_sk_2.png' % file_name)
        save_image((1 - gt_3[np.newaxis, :, :, np.newaxis]) * 255, true_img_sk_path_3, padding=0)
        extracted_array = self.extract(img) # img: 実線を含む入力画像  img2: 破線のみの入力画像
        # result_img_path = os.path.join(self.model_dir, "%s_1_result.png" % file_name)
        # save_image((1-(extracted_array>self.overlap_threshold))*255, result_img_path, padding=0)
        # result = self.calc_acc_of_SSaD(extracted_array, gt)

        # # モルフォロジー(CLOSE)
        # mor_img_path = os.path.join(self.model_dir, "%s_mor_in.png" % file_name)
        # save_image((extracted_array > self.overlap_threshold) * 255, mor_img_path, padding=0)
        # mor_img = cv2.imread("{}".format(mor_img_path), cv2.IMREAD_GRAYSCALE)
        # mor_img = cv2.resize(mor_img, (int(400), int(400)))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # extracted_array = cv2.morphologyEx(mor_img, cv2.MORPH_CLOSE, kernel)

        result, cf, re_f, xp = self.calc_acc_of_DtoS(extracted_array, gt, img) # img(実線込み) img2(破線のみ)

        n = 4 # 切り捨ての桁
        x = result["dash_to_solid"]
        y = math.floor(x * 10 ** n) / (10 ** n)
        if cf <= 1:
            z = "T"
        else:
            z = "F"

        # # モルフォロジー用
        # extracted_array = np.array(extracted_array)
        # max_intensity = np.amax(extracted_array)
        # extracted_array = (extracted_array > 0) * max_intensity
        # extracted_array = extracted_array / max_intensity

        result_img_path = os.path.join(self.model_dir, "%s_result_{}_{}_{}.png".format(z, y, cf) % file_name)
        save_image((1-(extracted_array>self.overlap_threshold))*255, result_img_path, padding=0)
        # save_image((1 - extracted_array[np.newaxis, :, :, np.newaxis]) * 255, result_img_path, padding=0) # モル

        result_img_path_2 = os.path.join(self.model_dir, "%s_result_2_{}_{}_{}.png".format(z, y, cf) % file_name)
        save_image((1 - xp[np.newaxis, :, :, np.newaxis]) * 255, result_img_path_2, padding=0)
        f_img_path = os.path.join(self.model_dir, "%s_N.over_{}.png".format(cf) % file_name)
        save_image((1 - re_f[np.newaxis, :, :, np.newaxis]) * 255, f_img_path, padding=0)

        #正解画像の成分数
        collect_data = Tikz(true_img_sk_path, height, width, True, 30)
        collect_components, collect_components_image = self.png_component(collect_data, 401, 401)

        self.save(collect_components_image, "%s/%s_1_single_components.png" % (self.model_dir, collect_data.fname),
                  is_put_text=True)

        #出力の成分数
        predict_data = Tikz(result_img_path, height, width, True, 30)
        png_components, predict_components_image = self.png_component(predict_data, 401, 401)

        self.save(predict_components_image, "%s/%s_1_single_components.png" % (self.model_dir, predict_data.fname),
                  is_put_text=True)

        return result, cf, collect_components, png_components

    def svg_component(self, file_path, width, height):
        with open(file_path, 'r') as f:
            svg = f.read()

        svg = svg.format(w=width, h=height)
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:, :, 3].astype(np.float)
        max_intensity = np.amax(s)

        s = (s > 30) * max_intensity
        s = s / max_intensity

        svg_xml = et.fromstring(svg)
        path_list = []
        num_paths = len(svg_xml[0])

        for i in range(1, num_paths):  # svgの情報から線分を1本ずつに分ける
            svg_xml = et.fromstring(svg)
            svg_xml[0][0] = svg_xml[0][i]
            del svg_xml[0][1:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            path = (np.array(y_img)[:, :, 3] > 30)
            path_list.append(path)

        ov = np.zeros([height, width], dtype=np.int)
        for i in range(num_paths - 2):
            for j in range(i + 1, num_paths - 1):
                intersect = np.logical_and(path_list[i], path_list[j])  # 各成分同士の交点を抽出
                ov = np.logical_or(intersect, ov)  # 交点
        kernel = np.ones((5, 5), np.uint8)
        iteration = self.rng.randint(1, 4)
        ov = cv2.dilate(ov.astype(np.uint8), kernel, iterations=iteration)  # 交点を膨張
        for i in range(10):
            path_id = self.rng.randint(0, num_paths - 1)
            y = path_list[path_id]
            y = np.logical_xor(y, np.logical_and(ov, y))  # 任意線分から交点を除去
            if len(np.nonzero(y)[0]) != 0:
                break
            y = path_list[path_id]

        n_labels, labels = cv2.connectedComponents(y.astype(np.uint8))  # 連結成分のラベリング

        return n_labels

    def png_component(self, data, width, height):
        single_components_list = []
        img = data.img
        kernel = np.ones((3, 3), np.uint8)
        ov_img = np.zeros([height, width])
        junc_img = self.extract_junction(img) #交点座標の取得
        junc_pixs = np.nonzero(junc_img)

        #patch_w, patch_h = int(self.patch_size / 2), int(self.patch_size / 2)
        patch_w, patch_h = int(32), int(32)

        for p in range(len(junc_pixs[0])):
            x, y = junc_pixs[0][p], junc_pixs[1][p]
            patch = self.get_patch(img, (x, y), (patch_h, patch_w))  # 交点領域32×32のパッチ画像の作成
            patch_center_pix = (int(patch_w / 2), int(patch_h / 2))
            m_initial, _ = cv2.connectedComponents(patch.astype(np.uint8))  # 連結成分のラベリング
            mask_img = np.zeros([patch_w, patch_h])
            mask_img[patch_center_pix[0], patch_center_pix[1]] = 1
            best_mask_img = mask_img
            for i in range(1, 7):
                # for i in range(1, 100):
                mask_img = cv2.dilate(mask_img.astype(np.uint8), kernel, iterations=i)  # 膨張処理
                rm_patch_img = np.logical_xor(patch, np.logical_and(patch, mask_img))  # patch画像の領域からmask画像の領域を取り除く
                m, _ = cv2.connectedComponents(rm_patch_img.astype(np.uint8))
                if m_initial >= m:
                    mask_img = np.logical_and(patch, mask_img)
                    continue
                else:
                    mask_img = np.logical_and(patch, mask_img)
                    best_mask_img = mask_img
                    m_initial = m
            junc_area_pixs = np.nonzero(np.logical_and(patch, best_mask_img))
            ov_img[junc_area_pixs[0] + (x - int(patch_w / 2)), junc_area_pixs[1] + (
                        y - int(patch_w / 2))] = 1  # 400×400のサイズに交点領域を作成
        s_components_img = np.logical_xor(img, ov_img)  # 交点領域の除去

        n_components, labels = cv2.connectedComponents(s_components_img.astype(np.uint8))  # 成分分割

        for i in range(1, n_components): #各成分の情報を取得
            pixs = np.nonzero(labels == i) #各成分ごとの座標を取得
            single_components_list.append(Component(pixs, label=self.label.get()))

        return n_components, single_components_list

    def get_patch(self, img, p, size=(64, 64)):
        w, h = size
        img = np.pad(img, (int(w / 2), int(h / 2)))
        p = (p[0] + int(w / 2), p[1] + int(h / 2))
        patch = img[p[0] - int(h / 2): p[0] + int(h / 2), p[1] - int(w / 2): p[1] + int(w / 2)]

        return patch

    def save(self, components_list, img_path, is_put_text=False):
        # Component型のリストを受け取り，各Componentごとに色分けした画像を保存する
        h = 401
        w = 401
        #save_img = np.zeros([self.height, self.width, 3])
        save_img = np.zeros([h, w, 3])
        for i in range(len(components_list)):
            component = components_list[i]
            pixs = component.pixs
            color_index = i % len(COLOR_LIST)
            save_img[pixs[0], pixs[1], :] = COLOR_LIST[color_index]
            #if is_put_text:
            #    cv2.putText(save_img, text="%d" % component.label, org=component.get_moment((h, w)),
            #                color=[255, 255, 255], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)
                #cv2.putText(save_img, text="%d" % component.label, org=component.get_moment((self.height, self.width)),
                #            color=[255, 255, 255], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)
        save_image(255 - save_img[np.newaxis, :, :, :], img_path, padding=0)

    def extract(self, img):
        x_batch = np.zeros([1, self.height, self.width, 1])
        x_batch[0, :, :, 0] = img

        if self.data_format == 'NCHW':
            x_batch = to_nchw_numpy(x_batch)
        y_b = self.s.run(self.y, feed_dict={self.x: x_batch})
        if self.data_format == 'NCHW':
            y_b = to_nhwc_numpy(y_b)
        return y_b

    def calc_acc_of_DtoS(self, extracted_array, gt_array, img2):
        # Separate Solid and Dashed
        extracted_array = extracted_array > self.overlap_threshold
        xp = extracted_array[0, :, :, 0]
        # xp = np.array(extracted_array) # モルフォロジー
        xp = xp.astype(np.float)
        # max_intensity = np.amax(xp) # モル
        # xp = (xp > 0) * max_intensity # モル
        # xp = xp / max_intensity # モル
        xp = xp - img2
        # xp = extracted_array
        # gt = gt_array[:, :, 0]
        gt = gt_array
        evaluation_DtoS, cf, re_f = calc_DtoS(xp, gt)
        # iou_of_DtoS = calc_IoU(xp > 0, gt > 0)
        print("Evaluation of dash to solid: {}".format(evaluation_DtoS))

        return {"dash_to_solid": evaluation_DtoS}, cf, re_f, xp

    def extract_junction(self, img): # 交点検出用　本研究では使用せず
        skel = skeletonize(img)
        junc_img = np.zeros((self.height, self.width))
        for col in range(1, self.height - 1):
            for row in range(1, self.width - 1):
                if not skel[col, row]:
                    continue

                n_changes = 0
                n_eight_neighbors = eight_neighbor(skel, [col, row])

                for i in range(8):
                    if n_eight_neighbors[i] != n_eight_neighbors[i + 1]:
                        n_changes += 1

                n_changes = n_changes / 2

                if n_changes >= 3:
                    junc_img[col, row] = 1

        if len(np.nonzero(junc_img)[0]) <= 1:
            return junc_img
        junc_img = self.clustering(junc_img)

        return junc_img

    def clustering(self, img: np.ndarray): # 交点検出用　本研究では使用せず
        pixs = np.nonzero(img)
        linkage_result = linkage(np.array(pixs).transpose(), method='ward', metric='euclidean')
        # plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
        # dendrogram(linkage_result)
        # plt.show()
        remove_index = []
        pixs_array = np.array(pixs).transpose()
        for one_link in linkage_result:
            if one_link[2] > 20.7:
                break
            p = calc_central_point(pixs_array[int(one_link[0])], pixs_array[int(one_link[1])])
            pixs_array = np.append(pixs_array, [p], axis=0)
            remove_index += [int(one_link[0])] + [int(one_link[1])]
        clustering_result = np.delete(pixs_array, remove_index, 0)
        result_pixs = tuple(clustering_result.transpose())
        array = np.zeros([self.height, self.width])
        array[result_pixs[0], result_pixs[1]] = 1
        return array

def calc_IoU(y, y_):
    y_I = np.logical_and(y > 0, y_ > 0)
    y_I_sum = np.sum(y_I)
    y_U = np.logical_or(y > 0, y_ > 0)
    y_U_sum = np.sum(y_U)

    # nonzero_id = np.where(y_U_sum != 0)[0]
    # if nonzero_id.shape[0] == 0:
    #     acc = 1.0
    # else:
    #     acc = np.average(y_I_sum[nonzero_id] / y_U_sum[nonzero_id])
    # iou += acc
    iou = y_I_sum / y_U_sum

    return iou

def calc_DtoS(xp, y):
    ev_y = xp - y
    cf = np.count_nonzero(ev_y < 0)
    ct = np.count_nonzero(y > 0) - cf
    if np.count_nonzero(y > 0) == 0:
        ev_DtoS = 1.0
    else:
        ev_DtoS = ct / np.count_nonzero(y > 0)
    # re_f = (ev_y > 0) * 0
    re_f = (ev_y < 0) * 1

    return ev_DtoS, cf, re_f

def eight_neighbor(array, target): # 交点検出用　本研究では使用せず
    '''
    :param array: image_array
    :param target: target pixel
    :return: list
     ┌─────────────────┐
　　　　│ 0,8 │  1  │  2  │
　　　　├─────┬─────┬─────┤
　　　　│  7  │ tar │  3  │
　　　　│─────┬─────┬─────┤
　　　　│  6  │  5  │  4  │
　　　　└─────┴─────┴─────┘
    '''
    return [
        array[target[0] - 1, target[1] - 1],
        array[target[0] - 1, target[1]],
        array[target[0] - 1, target[1] + 1],
        array[target[0], target[1] + 1],
        array[target[0] + 1, target[1] + 1],
        array[target[0] + 1, target[1]],
        array[target[0] + 1, target[1] - 1],
        array[target[0], target[1] - 1],
        array[target[0] - 1, target[1] - 1],
    ]

def calc_central_point(p1, p2): # 交点検出用
    return [int(round((p1[0] + p2[0]) / 2)), int(round((p1[1] + p2[1]) / 2))]

def main(config):
    from utils import prepare_dirs_and_logger, save_config
    prepare_dirs_and_logger(config)
    save_config(config)
    from data_line import BatchManager

    batch_manager = BatchManager(config)

    height = batch_manager.height
    width = batch_manager.width

    evaluater = Evaluater(config, batch_manager)
    model_log = evaluater.load_overlapnet
    evaluater.evaluate(height, width, model_log)

if __name__ == '__main__':
    from config import get_config
    config, unparsed = get_config()
    main(config)