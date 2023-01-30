import numpy as np
import cairosvg
import io
from PIL import Image
import xml.etree.ElementTree as et
import os
from components import *

class Data:
    # segmentation.pyを実行するために必要なデータ型である．svgデータは様々な表記方法があるため，適用したいデータに合わせたコードを書く必要がある．
    # Dataを親クラスとして継承し，適用したいデータに合わせたread_svg()メソッドを作成する
    # ラスタ画像の読み込み部分は未実装
    def __init__(self, fpath, w, h, is_binary, binary_threshold):
        self.fpath = fpath
        self.fext = os.path.splitext(self.fpath)[-1]
        self.fname = os.path.split(os.path.splitext(self.fpath)[0])[-1]
        self.width = w
        self.height = h
        self.is_binary = is_binary
        self.binary_threshold = binary_threshold
        if '.svg_pre' == self.fext:
            self.is_vector = True
            self.img, self.num_paths, self.path_list = self.read_svg()
        else:
            self.is_vector = False
            pass
        if '.bmp' == self.fext:
            self.is_vector = True
            self.img, self.num_paths, self.path_list = self.read_bmp()
        if '.png' == self.fext:
            self.is_vector = True
            self.img, self.num_paths, self.path_list = self.read_bmp()

    def read_svg(self):
        return np.ndarray, int(), list()


    def read_bmp(self):
        img_test = Image.open(self.fpath).convert('L')
        s = 255 - np.array(img_test)
        # import matplotlib.pyplot as plt
        # plt.imshow(s)
        # plt.show()
        print(np.unique(s)[-2])
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > self.binary_threshold) * max_intensity  # 2値化
        s = s / max_intensity
        # np.save('svg_array', s)

        return s, None, None

    def get_overlap(self):
        ov = np.zeros([self.height, self.width], dtype=np.int)
        for i in range(self.num_paths-1):
            for j in range(i + 1, self.num_paths):
                intersect = np.logical_and(self.path_list[i], self.path_list[j])
                ov = np.logical_or(intersect, ov)

        return ov

    def get_gt_basic_components(self, ov_components_list):
        ov_img = np.zeros([self.height, self.width])
        for ov_component in ov_components_list:
            ov_img[ov_component.pixs[0], ov_component.pixs[1]] = 1
        gt_basic_components_list = []
        for i in range(len(self.path_list)):
            basic_component_img = np.logical_xor(np.logical_and(self.path_list[i], ov_img), self.path_list[i])
            gt_basic_components_list.append(Component(np.nonzero(basic_component_img), label=0))

        return gt_basic_components_list

    def __str__(self):
        return "fname: {}".format(self.fpath)


class Tikz(Data):
    def __init__(self, fpath, w, h, is_binary, binary_threshold):
        super(Tikz, self).__init__(fpath, w, h, is_binary, binary_threshold)

    def read_svg(self):
        with open(self.fpath, 'r') as f:
            svg = f.read()

        svg_xml = et.fromstring(svg) #svgファイルの中身の参照
        svg_xml.attrib['width'] = "{w}".format(w=self.width)
        svg_xml.attrib['height'] = "{h}".format(h=self.height)
        svg = et.tostring(svg_xml)
        img = cairosvg.svg2png(bytestring=svg) #svg→png?
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > self.binary_threshold) * max_intensity  # 2値化
        s = s / max_intensity
        # np.save('svg_array', s)

        path_list = []
        svg_xml = et.fromstring(svg)
        print(svg_xml[1])
        num_paths = len(svg_xml[1])

        for i in range(0, num_paths):
            svg_xml = et.fromstring(svg)
            svg_xml[1][0] = svg_xml[1][i]
            del svg_xml[1][1:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            path = np.array(y_img)[:, :, 3]
            if self.is_binary:
                path = (path > self.binary_threshold) * max_intensity  # 二値化
            path_list.append(path)

        return s, num_paths, path_list


class FreeLine(Data):
    def __init__(self, fpath, w, h, is_binary, binary_threshold):
        super(FreeLine, self).__init__(fpath, w, h, is_binary, binary_threshold)

    def read_svg(self):
        with open(self.fpath, 'r') as f:
            svg = f.read()

        svg = svg.format(w=self.width, h=self.height)
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > self.binary_threshold) * max_intensity  # 2値化
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
            path = np.array(y_img)[:, :, 3]
            if self.is_binary:
                path = (path > self.binary_threshold) * max_intensity  # 二値化
            path_list.append(path)

        return s, num_paths, path_list


class Sketch(Data):
    def __init__(self, fpath, w, h, is_binary, binary_threshold):
        super(Sketch, self).__init__(fpath, w, h, is_binary, binary_threshold)

    def read_svg(self):
        with open(self.fpath, 'r') as f:
            svg = f.read()

        svg = svg.format(w=self.width, h=self.height)
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > self.binary_threshold) * max_intensity  # 2値化
        s = s / max_intensity
        # np.save('svg_array', s)

        path_list = []
        svg_xml = et.fromstring(svg)
        num_paths = len(svg_xml)

        for i in range(0, num_paths):
            svg_xml = et.fromstring(svg)
            svg_xml[0] = svg_xml[i]
            del svg_xml[1:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            path = np.array(y_img)[:, :, 3]
            if self.is_binary:
                path = (path > self.binary_threshold) * max_intensity  # 二値化
            path_list.append(path)

        return s, num_paths, path_list


class Data_line(Data):
    def __init__(self, fpath, w, h, is_binary, binary_threshold):
        super(Data_line, self).__init__(fpath, w, h, is_binary, binary_threshold)

    def read_svg(self):
        with open(self.fpath, 'r') as f:
            svg = f.read()


        svg = svg.format(w=self.width, h=self.height)
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
        max_intensity = np.amax(s)
        if self.is_binary:
            s = (s > self.binary_threshold) * max_intensity  # 2値化
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
            path = np.array(y_img)[:, :, 3]
            if self.is_binary:
                path = (path > self.binary_threshold) * max_intensity  # 二値化
            path_list.append(path)

        return s, num_paths, path_list


def main():
    f_path = r"C:\Users\user1\PycharmProjects\cnn_raster2vec\data\sketch_main\test\0.svg_pre"
    data_manager = Data_sketch(f_path, 400, 400, True, 30)
    pass


if __name__ == '__main__':
    main()

