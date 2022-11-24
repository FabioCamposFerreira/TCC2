import cv2 as cv
import numpy as np

import image_processing


def histogram_reduce(im):
    """Recude 256 histogram features to n_features"""
    n_features = 60
    hist = histogramFull(im)
    step = int(256/n_features)
    new_hist = []
    for index in range(n_features):
        new_hist += [sum(hist[step*index:(step*index)+step])]
    return new_hist

def histogramFull(im):
    """Receive path image and return histogram of the channel H"""
    im = image_processing.processing(im)
    h = np.squeeze(cv.calcHist([im], [0], None, [256], [0, 256])).tolist()
    return normalize(h)

def normalize(list_):
    """Normalize list or array"""
    x_max = max(list_)
    x_min = min(list_)
    difference = x_max-x_min
    if not difference:
        raise Exception("Extract feature is a string of zeros")
    return [(x-x_min)/(difference) for x in list_]


def get_pattern(im):
    return histogram_reduce(im)
