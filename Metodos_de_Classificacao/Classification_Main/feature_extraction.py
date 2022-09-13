# Author: Fábio Campos Ferreira
# Contains modules for extracting features from the images processed by the **image_processing.py** modules
# In general, the modules receive the images processed by the **feature_extraction.py** modules, extracted and treated features with their identification labels, are the input to the **training.py** modules
# When run as **main**, graphics and tables are generated to compare the features
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def histogram(im, library_img):
    """Receive image and return histogram of the channel H"""
    if library_img == "Pillow":
        return im.getchannel(channel=0).histogram(mask=None, extrema=None)
    elif library_img == "OpenCV":
        return np.squeeze(cv.calcHist([im], [0], None, [256], [0, 256])).tolist()


def get_features(im, feature, library_img):
    """Extract image features
    """
    if feature == "histogram":
        return histogram(im, library_img)
