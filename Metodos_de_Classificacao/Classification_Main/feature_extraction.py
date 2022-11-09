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


def image_contours(im, im_name: str, library_img):
    """Find contours in image"""
    if library_img == "Pillow":
        im = im.getchannel(channel=1)
        contours = cv.findContours(np.array(im))
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


def histogram_reduce(im, im_name: str, library_img, n_features: int):
    """Recude 256 histogram features to n_features"""
    hist = histogram(im, im_name, library_img)[0][1]
    step = int(256/n_features)
    new_hist = []
    for index in range(n_features):
        new_hist += [sum(hist[step*index:(step*index)+step])]
    return [[im_name, normalize(new_hist)]]


def histogram(im, im_name: str, library_img):
    """Receive image and return histogram of the channel H"""
    if library_img == "Pillow":
        return [[im_name, normalize(im.getchannel(channel=0).histogram(mask=None, extrema=None))]]
    elif library_img == "OpenCV":
        return [[im_name, normalize(np.squeeze(cv.calcHist([im], [0], None, [256], [0, 256])).tolist())]]


def histogram_filter(im, im_name: str, library_img: str):
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


def get_features(im, im_name, feature, library_img, n_features=10):
    """Extract image features
    """
    features = []
    if feature == "histogram":
        features = histogram(im, im_name, library_img)
    elif feature == "histogram_filter":
        features = histogram_filter(im, im_name, library_img)
    elif feature == "histogram_reduce":
        features = histogram_reduce(im, im_name, library_img, n_features)
    elif feature == "image_patches":
        features = image_patches(im, im_name, library_img, n_features)
    elif feature == "image_contours":
        features = image_contours(im, im_name, library_img)
    return features
