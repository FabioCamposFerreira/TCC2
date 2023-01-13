
import sys
import os
from bokeh.models import LabelSet, HoverTool, CustomJS, Dropdown
import itertools
import pandas as pd
from bokeh.transform import factor_cmap
from bokeh.palettes import Dark2_5 as palette
from bokeh.layouts import column, row
import time
import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter

from typing import List
from bokeh.models import CustomJS, Dropdown
import bokeh.plotting as bokeh
from skimage.feature import hog as hog_sk
sys.path.append('../../Metodos_de_Classificacao/Classification_Main/')


def gradienteHistogram(im, index):
    block_size = (2, 2)
    cell_size = (16, 16)
    nbins = 9
    hog = cv.HOGDescriptor(_winSize=(im.shape[1] // cell_size[1] * cell_size[1],
                                     im.shape[0] // cell_size[0] * cell_size[0]),
                           _blockSize=(block_size[1] * cell_size[1],
                                       block_size[0] * cell_size[0]),
                           _blockStride=(cell_size[1], cell_size[0]),
                           _cellSize=(cell_size[1], cell_size[0]),
                           _nbins=nbins)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    hist = hog.compute(im)
    cv.imwrite(str(index)+"hog.png", h)


def filterKmeans(im, k=10, index=0):
    Z = np.float32(im.reshape((-1, 3)))
    ret, label, center = cv.kmeans(Z, k, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                   10, cv.KMEANS_RANDOM_CENTERS)
    im = center[label.flatten()].reshape((im.shape))
    cv.imwrite("".join((str(index), "kmeans.png")), im)

def images_union2(ims: List[np.ndarray], blank_size=10):
    im_final1 = ims[0]
    im_final2 = ims[2]
    for index in range(1, 2):
        try:
            im_final1 = np.concatenate((im_final1, np.zeros((ims[0].shape[0], blank_size))+255, ims[index]), axis=1)
        except ValueError:
            im_final1 = np.concatenate((im_final1, np.zeros((ims[0].shape[0], blank_size, ims[0].shape[2]))+255,
                                        ims[index]), axis=1)
    for index in range(3, len(ims)):
        try:
            im_final2 = np.concatenate((im_final2, np.zeros((ims[0].shape[0], blank_size))+255,
                                        ims[index]), axis=1)
        except ValueError:
            im_final2 = np.concatenate((im_final2, np.zeros((ims[0].shape[0], blank_size, ims[0].shape[2]))+255,
                                        ims[index]), axis=1)
    try:
        im_final = np.concatenate((im_final1, np.zeros((blank_size, im_final2.shape[1]))+255, im_final2), axis=0)
    except ValueError:
        im_final = np.concatenate((im_final1, np.zeros((blank_size, im_final2.shape[1], im_final2.shape[2]))+255,
                                   im_final2), axis=0)
    cv.imwrite("images_merged.png", im_final)

def images_union(ims: List[np.ndarray], blank_size=10):
    im_final1 = ims[0]
    im_final2 = ims[3]
    for index in range(1, 3):
        try:
            im_final1 = np.concatenate((im_final1, np.zeros((ims[0].shape[0], blank_size))+255, ims[index]), axis=1)
        except ValueError:
            im_final1 = np.concatenate((im_final1, np.zeros((ims[0].shape[0], blank_size, ims[0].shape[2]))+255,
                                        ims[index]), axis=1)
    for index in range(4, len(ims)):
        try:
            im_final2 = np.concatenate((im_final2, np.zeros((ims[0].shape[0], blank_size))+255,
                                        ims[index]), axis=1)
        except ValueError:
            im_final2 = np.concatenate((im_final2, np.zeros((ims[0].shape[0], blank_size, ims[0].shape[2]))+255,
                                        ims[index]), axis=1)
    try:
        im_final = np.concatenate((im_final1, np.zeros((blank_size, im_final2.shape[1]))+255, im_final2), axis=0)
    except ValueError:
        im_final = np.concatenate((im_final1, np.zeros((blank_size, im_final2.shape[1], im_final2.shape[2]))+255,
                                   im_final2), axis=0)
    cv.imwrite("images_merged.png", im_final)


def feature_sift(arq: str, feature: str, library_img: str, n_features: int, inverted: bool):
    im = image_processing.img_process(arq, "OpenCV", "_x", inverted=False)
    gray = image_processing.img_process(arq, "OpenCV", "gray_x", inverted=False)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    cv.imwrite(str(time.time())+".png", cv.drawKeypoints(gray, kp, im, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    # _, _, center = cv.kmeans(np.float32(des), 60, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
    #                         10, cv.KMEANS_RANDOM_CENTERS)


def add_hue_bar(f, length: int):
    """Add in graphic from bokeh bar with hue spectrum"""
    x = [r for r in range(length)]
    y = [-1 for _ in range(length)]
    hsv = [np.uint8([[[hue, 255, int(255/2)]]]) for hue in range(0, 256, int(256/length))]
    rgb = [cv.cvtColor(hsv_one, cv.COLOR_HSV2RGB_FULL)[0][0] for hsv_one in hsv]
    hex_ = ['#%02x%02x%02x' % tuple(rgb_one.tolist()) for rgb_one in rgb]
    f.square(x, y, size=20, color=hex_)
    return f


def graphics_interactive(curves: list, labels: list, file_path: str):
    """Save in file html graph interactive with many curves"""
    colors = itertools.cycle(palette)
    length = len(curves[0])
    x = range(length)
    f = bokeh.figure(sizing_mode="stretch_both", tools="pan,wheel_zoom,box_zoom,reset,save", output_backend="svg")
    for curve, label, color in zip(curves, labels, colors):
        l = f.line(x, curve,  line_color=color, legend_label=label, line_width=2)
        f.add_tools(HoverTool(renderers=[l], tooltips=[('Name', label), ]))
    f.legend.location = "top_right"
    f.legend.click_policy = "hide"
    f = add_hue_bar(f, length)
    bokeh.output_file(file_path+".html")
    bokeh.save(f)


def optimization_graph(points: dict, file_path: str):
    """Save graph 2d with points and one legend to each point

    Parameters
    ----------
    points : dict
        keys are accuracy and the method parameters
    """
    menu = []
    labels = ""
    points["x"] = points["accuracy"]
    points["y"] = points["accuracy"]
    key_strings = ""
    for key in points.keys():
        if type(points[key][0]) != str:
            if key != "x" and key != "y":
                menu += [(key, key)]
        else:
            key_strings = key
        labels += "".join((",", key, ":@", key))
    source = pd.DataFrame.from_dict(points)
    index_cmap = factor_cmap(key_strings, palette=palette, factors=sorted(source[key_strings].unique()))
    source = bokeh.ColumnDataSource(points)
    TOOLTIPS = [("(x,y)", "($x, $y)"), ("label", labels)]
    f = bokeh.figure(sizing_mode="stretch_both", output_backend="svg", tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     tooltips=TOOLTIPS)
    f.circle("x", "y", source=source, legend_field="kernel", size=10, color=index_cmap)
    f.legend.click_policy = "hide"
    f.add_layout(f.legend[0], 'right')
    f.xaxis.axis_label = "accuracy"
    f.yaxis.axis_label = "accuracy"
    dropdown_x = Dropdown(label="Select X", button_type="warning", menu=menu)
    dropdown_y = Dropdown(label="Select Y", button_type="warning", menu=menu)
    dropdown_x.js_on_event("menu_item_click", CustomJS(args=dict(source=source, x_label=f.xaxis[0]), code="""
    const new_data = Object.assign({}, source.data)
    new_data.x = source.data[this.item]
    source.data = new_data
    console.log(x_label.axis_label)
    x_label.axis_label = this.item
    """))
    dropdown_y.js_on_event("menu_item_click", CustomJS(args=dict(source=source, y_label=f.yaxis[0]), code="""
    const new_data = Object.assign({}, source.data)
    new_data.y = source.data[this.item]
    source.data = new_data
    y_label.axis_label = this.item
    """))
    bokeh.output_file(file_path+'.html')
    bokeh.show(column(row(dropdown_x, dropdown_y, sizing_mode="scale_width"), f, sizing_mode="stretch_both"))


def image_contours(im, im_name: str, library_img):
    """Find contours in image"""
    curves = np.zeros((1, 255), dtype=int)
    labels = []
    if library_img == "OpenCV":
        contours, _ = cv.findContours(im, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        im_color_c = img_process(arq, library_img="OpenCV", img_processing=[])
        cv.drawContours(im_color_c, contours, -1, (0, 255, 0), -1)
        cv.imwrite("im_color_c.png", im_color_c)
        if len(contours):
            for index, contour in enumerate(contours):
                if cv.contourArea(contour) > 3e3:
                    im_color_v = img_process(arq, library_img="OpenCV", img_processing=["gaussian", "HSV"])
                    im_color = img_process(arq, library_img="OpenCV", img_processing=["HSV", "get_0", "gaussian"])
                    cv.imwrite("h.png", im_color)
                    mask = np.zeros(im.shape, dtype="uint8")
                    cv.drawContours(mask, [contour], -1, 255, -1)
                    im_color[mask == 0] = 0
                    im_color_v[mask == 0] = (0, 0, 0)
                    #im_color_v[:,:,1] = 255
                    #im_color_v[:,:,2] = 255/2
                    # cv.imwrite(str(index)+".png",im_color)
                    cv.imwrite(str(index)+"v.png", cv.cvtColor(im_color_v, cv.COLOR_HSV2BGR_FULL))
                    curves = np.vstack(
                        (curves, np.squeeze(normalize(cv.calcHist([im_color], [0], None, [256], [0, 256])[1:]))))
                    labels += [str(index)]
    graphics_interactive(curves[1:], labels, "temp")


def image_patches(im, im_name: str, library_img, patches_len=128):
    """Split image  in paths of 256 pixels"""
    patches = []
    step = patches_len
    left, upper, right, lower = 0, 0, step, step
    count = 0
    if library_img == "Pillow":
        l, h = im.size
    if library_img == "OpenCV":
        h, l = im.shape[:2]
        count = 0
        while right < l:
            while lower < h:
                if library_img == "Pillow":
                    a = im.crop((left, upper, right, lower))
                if library_img == "OpenCV":
                    a = im[upper:lower, left:right]
                patches.append([im_name+str(count), np.hstack(np.array(a))])
                count += 1
                left, upper, right, lower = left, upper+step, right, lower+step
            left, upper, right, lower = left+step, 0, right+step, step
    return patches


def histogram_reduce(im, im_name: str, library_img, n_features: int):
    """Recude 256 histogram features to n_features"""
    hist = histogram(im, im_name, library_img)[0][1]
    step = int(256/n_features)
    new_hist = []
    for index in range(n_features):
        new_hist += [sum(hist[step*index:(step*index)+step])]
    return [[im_name, normalize(new_hist)]]


def normalize(list_):
    """Normalize list or array"""
    x_max = max(list_)
    x_min = min(list_)
    difference = x_max-x_min
    if not difference:
        raise Exception("Extract feature is a string of zeros")
    return [(x-x_min)/(difference)*100 for x in list_]


def histogram_filter(im, im_name: str, library_img: str):
    """Receive image and return histogram of the channel H excruing pixels with low saturation and value in extrems"""
    if library_img == "Pillow":
        im = np.array(im)
        im = np.vstack(im)
    saturation_tolerance = 0.8
    value_tolerance = 0.3
    temp = ((im[:, :, 1] > 255-255*saturation_tolerance)
            & (im[:, :, 2] > 255/2-255/2*value_tolerance) & (im[:, :, 2] < 255/2+255/2*value_tolerance))
    im_rgb = open_image(arq, library_img)

    # im_rgb[temp] = (0, 255, 0)
    temp = np.array([not i for i in np.nditer(temp)]).reshape(im_rgb.shape[:2])
    im_rgb[temp] = (0, 0, 0)
    cv.imwrite("filtrada.png", im_rgb)
    im = im[temp]
    # im = im[(im[:,:, 1] > 255-255*saturation_tolerance)
    # & (im[:,:, 2] > 255/2-255/2*value_tolerance) & (im[:,:, 2] < 255/2+255/2*value_tolerance)]
    return [[im_name, normalize(np.histogram(im[:, 0], bins=range(256+1))[0])]]


def histogram(im, im_name: str, library_img):
    """Receive image and return histogram of the channel H"""
    if library_img == "Pillow":
        return [[im_name, normalize(im.getchannel(channel=0).histogram(mask=None, extrema=None))]]
    elif library_img == "OpenCV":
        return [[im_name, normalize(np.squeeze(cv.calcHist([im], [0], None, [256], [0, 256])).tolist())]]


def open_image(arq, library_img, inverted=False):
    """Get a path if the image and return it as pillow/array Image"""
    if library_img == "Pillow":
        im = Image.open(arq)
        # rotate to lay down the image
        l, h = im.size
        if l < h:
            im = im.rotate(angle=90, resample=0, expand=True)
        if inverted == True:
            im = im.rotate(angle=180, resample=0, expand=True)
        return im.resize(constants.RESOLUTION, resample=Image.Resampling.NEAREST)
    elif library_img == "OpenCV":
        # TODO: its not working, conversion bug color
        im = cv.imread(arq)
        h, l = im.shape[:2]
        if l < h:
            im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
        if inverted == True:
            im = cv.rotate(im, cv.ROTATE_180)
        return cv.resize(im, constants.RESOLUTION, cv.INTER_NEAREST)


def img_process(arq, library_img, img_processing: List[str], inverted=False):
    "Get a path if the image, process and return it as pillow/array Image"
    im = open_image(arq, library_img, inverted=False)
    """ (B, G, R) = cv.split(im)
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0
    im = cv.merge([B, G, R]) """
    cv.imwrite("original.png", im)
    im = processing(im, library_img, img_processing)
    return im


def temp1():
    im = img_process(arq, library_img="OpenCV", img_processing=[
                     "gray", "histogram_equalization", "gaussian", "canny", "thresh", "morphology"])
    cv.imwrite("processada.png", im)
    im = img_process(arq, library_img="OpenCV", img_processing=[
                     "gaussian", "HSV", "get_0", "sobel", "thresh", "morphology"])
    y = histogram_reduce(im, im_name="test", library_img="OpenCV", n_features=10)[0][1]
    y = image_patches(im, im_name="test", library_img="OpenCV")[0][1]
    image_contours(im, im_name="test", library_img="OpenCV")
    f = bokeh.figure(sizing_mode="stretch_both", output_backend="svg", tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     )
    f.line(range(len(y)), y)
    bokeh.output_file("temp.html")
    bokeh.show(f)


def temp2():
    " Merge images reference"
    path = "../../Data_Base/Data_Base_Refencia"
    data_base = os.listdir(path)
    data_base.sort(key=others.images_sort)
    for index in range(len(data_base)):
        data_base[index] = "".join((path, "/", data_base[index]))
    ims = []
    for index, arq in enumerate(data_base):
        ims.append(open_image(arq, "OpenCV", inverted=False))
    images_union(ims)


def temp3():
    "Process image and merge results"
    path = "../../Data_Base/Data_Base_Refencia"
    data_base = os.listdir(path)
    data_base.sort(key=others.images_sort)
    for index in range(len(data_base)):
        data_base[index] = "".join((path, "/", data_base[index]))
    ims = []
    for path in data_base:
        ims.append(image_processing.img_process(path, "OpenCV", "gray_filterGaussianBlur_5_x", inverted=False)[0])
    images_union(ims)


def temp4():
    "Test SIFT"
    path = "../../Data_Base/Data_Base_Refencia"
    data_base = os.listdir(path)
    data_base.sort(key=others.images_sort)
    for index in range(len(data_base)):
        data_base[index] = "".join((path, "/", data_base[index]))
    ims = []
    for index, path in enumerate(data_base):
        feature_sift(path, "", "", 0, False)


def temp5():
    "Test filter kmeans"
    path = "../../Data_Base/Data_Base_Refencia"
    data_base = os.listdir(path)
    data_base.sort(key=others.images_sort)
    for index in range(len(data_base)):
        data_base[index] = "".join((path, "/", data_base[index]))
    ims = []
    for index, path in enumerate(data_base):
        im = image_processing.img_process(path, "OpenCV", "filterGaussianBlur_25_x", inverted=False)
        filterKmeans(im, 2, index)


def temp6():
    "Test HOG"
    path = "../../Data_Base/Data_Base_Refencia"
    data_base = os.listdir(path)
    data_base.sort(key=others.images_sort)
    for index in range(len(data_base)):
        data_base[index] = "".join((path, "/", data_base[index]))
    for index, path in enumerate(data_base):
        im = image_processing.img_process(path, "OpenCV", "gray_thresh_x", inverted=False)
        gradienteHistogram(im, index)


def temp7():
    "Test Bilateral"
    path = "../../Data_Base/Data_Base_Refencia"
    data_base = os.listdir(path)
    data_base.sort(key=others.images_sort)
    for index in range(len(data_base)):
        data_base[index] = "".join((path, "/", data_base[index]))
    for index, path in enumerate(data_base):
        im = image_processing.img_process(path, "OpenCV", "filterBilateralFilter_x", inverted=False)
        cv.imwrite(str(index)+"bilateral.png", im)

def temp8():
    "Merge imagens"
    data_base=["2.png","5.png","10.png","20.png","50.png","100.png"]
    # data_base=["2.png","10.png","20.png","50.png"]
    ims = []
    for path in data_base:
        ims.append(image_processing.img_process(path, "OpenCV", "filterGaussianBlur_1_x", inverted=False)[0])
    images_union(ims)
    # images_union2('ims)'

def temp9():
    "sift imagens"
    data_base=["2.jpg"]
    # data_base=["2.png","10.png","20.png","50.png"]
    ims = []
    for path in data_base:
        ims.append(image_processing.img_process(path, "OpenCV", "gray_filterGaussianBlur_5_x", inverted=False)[0])
    im = ims[0]
    # from skimage.feature import SIFT
    sift = cv.SIFT_create()
    kp = sift.detect(im,None)
    im=cv.drawKeypoints(im,kp,im,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite("temp.png",im)

if __name__ == "__main__":
    import others
    import image_processing
    import constants
    temp9()
