# Author: Fábio Campos Ferreira
# Contain modules for image processing
# In general, the modules open the images from the database, process the images, the result is sent to be used by the modules of feature_extraction.py
from typing import List

import cv2 as cv
from PIL import Image, ImageFilter

import constants


def processing(im, library_img: str, img_processing: List[str]):
    """Make processing image"""
    if library_img == "Pillow":
        if "HSV" in img_processing:
            im = im.convert(mode="HSV")
        if "get_H" in img_processing:
            im = im.getchannel(0)
        if "filter_blur" in img_processing:
            im = im.filter(ImageFilter.BLUR)
        if "filter_contour" in img_processing:
            im = im.filter(ImageFilter.CONTOUR)
        if "filter_detail" in img_processing:
            im = im.filter(ImageFilter.DETAIL)
        if "filter_edgeEnhance" in img_processing:
            im = im.filter(ImageFilter.EDGE_ENHANCE)
        if "filter_edgeEnhanceMore" in img_processing:
            im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
        if "filter_emboss" in img_processing:
            im = im.filter(ImageFilter.EMBOSS)
        if "filter_findEdges" in img_processing:
            im = im.filter(ImageFilter.FIND_EDGES)
        if "filter_sharpen" in img_processing:
            im = im.filter(ImageFilter.SHARPEN)
        if "filter_smooth" in img_processing:
            im = im.filter(ImageFilter.SMOOTH)
        if "filter_smoothMore" in img_processing:
            im = im.filter(ImageFilter.SMOOTH_MORE)
    if library_img == "OpenCV":
        if "HSV" in img_processing:
            im = cv.cvtColor(im, cv.COLOR_BGR2HSV_FULL)
        if "get_H" in img_processing:
            im = im[:, :, 0]
        if "filter_blur" in img_processing:
            im = cv.blur(im, (5, 5))
        if "filter_median_blur" in img_processing:
            im = cv.medianBlur(im, 5)
        if "filter_gaussian_blur" in img_processing:
            im = cv.GaussianBlur(im, (5, 5), 0)
        if "filter_bilateral_filter" in img_processing:
            im = cv.bilateralFilter(im, 9, 75, 75)
    return im


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
    im = processing(im, library_img, img_processing)
    return im
