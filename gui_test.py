import io
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf
import models
from datetime import datetime
from ops import *

def __init__(self, main):
    #ファイル削除処理
    self.file_del()



#ウインドウを作成
win = tk.Tk()
win.title("破線の実線化")#タイトル
win.geometry("1500x600")#サイズ

#パーツを配置
#ラベル1を作成
label1 = tk.Label(text='■画像読込')
label1.place(x=10, y=10)
#ラベル2を作成
label2 = tk.Label(text='ファイルパス：')
label2.place(x=10, y=40)
#ラベル3
label3 = tk.Label(text='入力', font=("MSゴシック", "20"))
label3.place(x=200, y=470)
#ラベル4
label4 = tk.Label(text='出力', font=("MSゴシック", "20"))
label4.place(x=710, y=470)
#ラベル5
label5 = tk.Label(text='交点', font=("MSゴシック", "20"))
label5.place(x=1210, y=470)
#ファイルパスの表示欄を作成
input_box1 = tk.Entry(width=50)
input_box1.place(x=80, y=40)



#参照ボタンの動作
def button1_clicked(self):
    #ファイルパスを取得
    idir = r'C:\\descktop'
    filepath = tk.filedialog.askdirectory(initialdir = idir)


def file_select():
    input_box1.delete(0, tk.END)
    #ファイルパスを表示欄に表示
    idir = r'C:\\descktop'
    filetype = [("すべて", "*")]
    filepath = tk.filedialog.askopenfilename(filetypes = filetype, initialdir = idir)
    input_box1.insert(tk.END, filepath)

    # 選択されたファイルの画像を表示
    if filepath and os.path.isfile(filepath):
        # PILで画像を開きグレイスケールに変換
        img = Image.open(filepath).convert("L")
        cvr.image = img

        image = ImageTk.PhotoImage(image=img)
        cv.image = image  # ※ 重要! 何処かに参照を残す。

        # 表示位置を調整 (画像の中心が指定の座標に来る)
        cv.create_image(image.width() / 2, image.height() / 2, image=image)

def predict():
    data_format = "NCHW"
    conv_hidden_num = 64
    repeat_num = 20
    use_norm = True
    load_model = r"C:\Users\Shoya Nagata\PycharmProjects\vectornet_for_debug\log\overlap\line_1011_065125_test"
    imagepath = input_box1.get()
    img = Image.open(imagepath).convert("L")
    s = 255 - np.array(img)
    height, width = s.shape
    max_intensity = np.amax(s)
    # s = (s > 50) * max_intensity
    s = cv2.GaussianBlur(s, (5, 5), 0)
    s = cv2.threshold(s, 0, max_intensity, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    s = s[1]
    s = s / max_intensity


    pathnet_graph = tf.Graph()
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=tf.GPUOptions(allow_growth=True))
    sp = tf.Session(config=sess_config, graph=pathnet_graph)
    with pathnet_graph.as_default():
        xp = tf.placeholder(tf.float32, shape=[None, height, width, 1])
        if data_format == 'NCHW':
            xp = nhwc_to_nchw(xp)

        yp, _ = models.VDSR(xp, conv_hidden_num, repeat_num,
                            data_format, use_norm, train=False)
        show_all_variables()

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(load_model)

        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sp, os.path.join(load_model, ckpt_name))
        print('%s: Pre-trained model restored from %s' % (datetime.now(), load_model))

    xt = np.expand_dims(np.expand_dims(s, axis=0), axis=0)
    out = sp.run(yp, {xp: xt})
    out = (out > 0.5) * 255
    resultimg = 255 - out[0, 0, :, :]
    resultimg = Image.fromarray(resultimg.astype(np.uint8))
    img_len = len(os.listdir(r"C:\Users\Shoya Nagata\PycharmProjects\vectornet_for_debug\gui_result"))
    resultimg.save(r"C:\Users\Shoya Nagata\PycharmProjects\vectornet_for_debug\gui_result\{}.png".format(str(img_len).zfill(3)))
    image = ImageTk.PhotoImage(image=resultimg)
    cvr.image = image  # ※ 重要! 何処かに参照を残す。
    # 表示位置を調整 (画像の中心が指定の座標に来る)
    cvr.create_image(image.width() / 2, image.height() / 2, image=image)

def intersection():
    img_len = len(os.listdir(r"C:\Users\Shoya Nagata\PycharmProjects\vectornet_for_debug\gui_result"))
    img = cv2.imread(r"C:\Users\Shoya Nagata\PycharmProjects\vectornet_for_debug\gui_result\{}.png".format(str(img_len - 1).zfill(3)))
    # img = cvr.image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 8, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    image = Image.fromarray(img.astype(np.uint8))
    image = ImageTk.PhotoImage(image=image)
    cvi.image = image
    cvi.create_image(image.width() / 2, image.height() / 2, image=image)

#Canvasの作成
cv = tk.Canvas(win, width=400-1, height=400-1, bg="white",)
cv.place(x=20, y=70)

cvr = tk.Canvas(win, width=400-1, height=400-1, bg="white",)
cvr.place(x=520, y=70)

cvi = tk.Canvas(win, width=400-1, height=400-1, bg="white",)
cvi.place(x=1020, y=70)

#参照ボタン1を作成
button1 = tk.Button(text="参照", command=file_select, width=8)
button1.place(x=400, y=40)

#閉じるボタンを作成
button2 = tk.Button(text="閉じる", command=win.destroy, width=8)
button2.place(x=1400, y=550)

#実線化ボタン
button3 = tk.Button(text="実線化", command=predict, width=8)
button3.place(x=435, y=250)

#交点検出ボタン
button3 = tk.Button(text="交点検出", command=intersection, width=8)
button3.place(x=935, y=250)

#ウインドウを動かす
win.mainloop()