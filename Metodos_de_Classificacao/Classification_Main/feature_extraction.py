# Author: Fábio Campos Ferreira
# Contains modules for extracting features from the images processed by the **image_processing.py** modules
# In general, the modules receive the images processed by the **feature_extraction.py** modules, extracted and treated features with their identification labels, are the input to the **training.py** modules
# When run as **main**, graphics and tables are generated to compare the features
import cv2 as cv
import numpy as np


def histogram(im, library_img):
    """Receive image and return histogram of the channel H"""
    if library_img == "Pillow":
        return im.getchannel(channel=0).histogram(mask=None, extrema=None)
    elif library_img == "OpenCV":
        return np.squeeze(cv.calcHist([im], [0], None, [256], [0, 256])).tolist()


def histogram_filter(im, library_img: str):
    """Receive image and return histogram of the channel H excruing pixels with low saturation and value in extrems"""
    if library_img == "Pillow":
        saturation_tolerance = 0.5
        value_tolerance = 0.3
        im = np.array(im)
        im = np.vstack(im)
        im = im[(im[:, 1] > 255-255*saturation_tolerance) & (im[:, 2] > 255/2-255/2*value_tolerance) &
                (im[:, 2] < 255/2+255/2*value_tolerance)]
        return np.histogram(im[:, 0], bins=range(256))[0]


def normalize(list_):
    """Normalize list or array"""
    x_max = max(list_)
    x_min = min(list_)
    difference = x_max-x_min
    return [(x-x_min)/(difference) if difference else 0 for x in list_]


def get_features(im, feature, library_img):
    """Extract image features
    """
    if feature == "histogram":
        return (histogram(im, library_img))
    if feature == "histogram_filter":
        return normalize(histogram_filter(im, library_img))
