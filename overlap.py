def preprocess_overlap(file_path, w, h, rng, is_binary):
    with open(file_path, 'r') as f:
        svg = f.read()
    svg = svg.format(w=w, h=h)
    img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(img))
    s = np.array(img)[:, :, 3].astype(np.float)  # / 255.0
    max_intensity = np.amax(s)
    if is_binary:
       s = (s > 0) * max_intensity  # 2値化
    s = s / max_intensity

    # while True:
    path_list = []  # data_line ori
    svg_xml = et.fromstring(svg)
    num_paths = len(svg_xml[0])

    for i in range(0, num_paths):  # data_line_2 st
        if "stroke-dasharray" in svg_xml[0][i].attrib.keys():  # 破線を実線に変える
            del svg_xml[0][i].attrib["stroke-dasharray"]
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
    path = (np.array(y_img)[:, :, 3] > 0)  # data_line_2 end

    # for i in range(num_paths): # data_line ori st
    #     svg_xml = et.fromstring(svg)
    #     if "stroke-dasharray" in svg_xml[0][i].attrib.keys():
    #         del svg_xml[0][i].attrib["stroke-dasharray"]
    #     svg_xml[0][0] = svg_xml[0][i]
    #     del svg_xml[0][1:]
    #     svg_one = et.tostring(svg_xml, method='xml')

    #     # leave only one path
    #     y_png = cairosvg.svg2png(bytestring=svg_one)
    #     y_img = Image.open(io.BytesIO(y_png))
    #     path = (np.array(y_img)[:,:,3] > 0)
    #     path_list.append(path)

    # y = np.zeros([h, w], dtype=np.int)
    # for i in range(num_paths-1):
    #     for j in range(i+1, num_paths):
    #         intersect = np.logical_and(path_list[i], path_list[j])
    #         y = np.logical_or(intersect, y) # data_line ori end

    x = np.expand_dims(s, axis=-1)
    y = np.reshape(y, [h, w, 1])
    # y = np.expand_dims(y, axis=-1) # data_line ori
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
