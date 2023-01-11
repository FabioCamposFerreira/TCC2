import cv2 as cv
import numpy as np

import image_processing




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
    return [(x-x_min)/(difference)*100 for x in list_]


def get_pattern(im):
    return histogramFull(im)
