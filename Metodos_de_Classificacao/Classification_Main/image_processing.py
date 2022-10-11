# Author: Fábio Campos Ferreira
# Contain modules for image processing
# In general, the modules open the images from the database, process the images, the result is sent to be used by the modules of feature_extraction.py
from PIL import Image
import cv2 as cv


def open_image(arq, library_img, inverted=False):
    """Get a path if the image and return it as pillow Image"""
    if library_img == "Pillow":
        im = Image.open(arq).convert(mode='HSV', palette=0)
        # rotate to lay down the image
        l, h = im.size
        if l < h:
            im = im.rotate(angle=90, resample=0, expand=True)
        if inverted == True:
            im = im.rotate(angle=180, resample=0, expand=True)
        return im.resize((854, 480), resample=Image.Resampling.NEAREST)
    elif library_img == "OpenCV":
        # TODO: its not working, conversion bug color
        im = cv.cvtColor(cv.imread(arq), cv.COLOR_RGB2HSV_FULL)
        h, l = im.shape[:2]
        if l < h:
            im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
        if inverted == True:
            im = cv.rotate(im, cv.ROTATE_180)
        return cv.resize(im, (854, 480), cv.INTER_CUBIC)
