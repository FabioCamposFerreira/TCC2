# Author: Fábio Campos Ferreira
# Contains modules for extracting features from the images processed by the **image_processing.py** modules
# In general, the modules receive the images processed by the **feature_extraction.py** modules, extracted and treated features with their identification labels, are the input to the **training.py** modules
# When run as **main**, graphics and tables are generated to compare the features
from typing import List

from PIL import Image
import numpy as np
import cv2 as cv

import constants
import image_processing


def color_contours(arq: str, feature: str, library_img: str, inverted: bool):
    """Find contours in image"""
    im = image_processing.img_process(arq, library_img, feature.split("processingBreak")[0], inverted)
    im2 = image_processing.img_process(arq, library_img, feature.split("processingBreak")[1], inverted)
    im_name = "-".join((arq.split("/")[-1], "Inverted"*inverted))
    returns = []
    if library_img == "OpenCV":
        contours, _ = cv.findContours(im, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        if len(contours):
            for index, contour in enumerate(contours):
                if cv.contourArea(contour) > constants.AREA_MIN:
                    mask = np.zeros(im.shape, dtype="uint8")
                    cv.drawContours(mask, [contour], -1, 255, -1)
                    temp = np.array(im2)
                    temp[mask == 0] = 0
                    cv.imwrite("".join(("./Output/imgs/",im_name)),temp)
                    returns.append(["-".join((im_name, str(index))),
                                    np.squeeze(normalize(cv.calcHist([temp], [0], None, [256], [0, 256])[1:]))])
    return returns


def image_contours(im, im_name: str, library_img):
    """Find contours in image"""
    if library_img == "Pillow":
        im = im.getchannel(channel=1)
        contours = cv.findContours(np.array(im), mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        counts = 0
        returns = []
        for imgs in contours:
            returns.append([im_name+str(counts), imgs.resize((854, 480),
                           resample=Image.Resampling.NEAREST).histogram()])
            counts += 1
        return returns


def image_patches(im, im_name: str, library_img, patches_len: int):
    """Split image  in paths of 256 pixels"""
    patches = []
    step = int(patches_len**(1/2))
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


def histogram_reduce(arq: str, feature: str, library_img: str,n_features:int, inverted: bool):
    """Recude 256 histogram features to n_features"""
    im_name = "-".join((arq.split("/")[-1], "Inverted"*inverted))
    hist = histogramFull(arq, feature, library_img,inverted)[0][1]
    step = int(256/n_features)
    new_hist = []
    for index in range(n_features):
        new_hist += [sum(hist[step*index:(step*index)+step])]
    return [[im_name, normalize(new_hist)]]


def histogramFull(arq: str, feature: str, library_img: str, inverted: bool):
    """Receive path image and return histogram of the channel H"""
    im = image_processing.img_process(arq, library_img, feature, inverted)
    im_name = "-".join((arq.split("/")[-1], "Inverted"*inverted))
    h = []
    if library_img == "Pillow":
        h = im.getchannel(channel=0).histogram(mask=None, extrema=None)
    elif library_img == "OpenCV":
        h = np.squeeze(cv.calcHist([im], [0], None, [256], [0, 256])).tolist()
    return [[im_name, normalize(h)]]


def histogram_filter(im: np.ndarray, im_name: str, library_img: str):
    """Receive image and return histogram of the channel H excruing pixels with low saturation and value in extrems"""
    if library_img == "Pillow":
        im = np.array(im)
        im = np.vstack(im)
    im = im[(im[:, :, 1] > 255 - 255 * constants.SATURATION_TOLERANCE)
            & (im[:, :, 2] > 255 / 2 - 255 / 2 * constants.VALUE_TOLERANCE)
            & (im[:, :, 2] < 255 / 2 + 255 / 2 * constants.VALUE_TOLERANCE)]
    return [[im_name, np.histogram(im[:, 0], bins=range(256+1))[0]]]


def normalize(list_):
    """Normalize list or array"""
    x_max = max(list_)
    x_min = min(list_)
    difference = x_max-x_min
    if not difference:
        raise Exception("Extract feature is a string of zeros")
    return [(x-x_min)/(difference)*100 for x in list_]


def get_features(arq: str, feature, library_img, n_features=10, inverted=False):
    """Extract image features
    """
    if "histogramFull" in feature:
        features = histogramFull(arq, feature, library_img, inverted)
    elif "histogramFilter" in feature:
        features = histogram_filter(arq, feature, library_img, inverted)
    elif "histogramReduce" in feature:
        features = histogram_reduce(arq, feature, library_img, n_features, inverted)
    elif "imagePatches" in feature:
        features = image_patches(arq, feature, library_img, n_features, inverted)
    elif "imageContours" in feature:
        features = image_contours(arq, feature, library_img, inverted)
    elif "colorContours" in feature:
        features = color_contours(arq, feature, library_img, inverted)
    else:
        Exception("Feature not found!")
    return features
