# Author: Fábio Campos Ferreira
# Contain modules for image processing
# In general, the modules open the images from the database, process the images, the result is sent to be used by the modules of feature_extraction.py
from typing import List
import warnings

import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter

import constants

def image2contours(im, library_img):
    """Find contours"""
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
                    returns.append(temp)
    return returns

def image2patches(im, library_img, patches_len: int):
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
                patches.append([np.hstack(np.array(a))])
                count += 1
                left, upper, right, lower = left, upper+step, right, lower+step
            left, upper, right, lower = left+step, 0, right+step, step
    return patches

def image2images(im, library_img: str, slip_criteria: str):
    """Split image in masks with criteria"""
    if library_img == "OpenCV":
        if "patches" == processing:
            image2patches(im, library_img, patches_len)
def processing(im, library_img: str, img_processing: List[str]):
    """Make processing image"""
    for index, processing in enumerate(img_processing):
        if library_img == "Pillow":
            if "HSV" in processing:
                im = im.convert(mode="HSV")
            if "get_H" in processing:
                im = im.getchannel(0)
            if "filter_blur" in processing:
                im = im.filter(ImageFilter.BLUR)
            if "filter_contour" in processing:
                im = im.filter(ImageFilter.CONTOUR)
            if "filter_detail" in processing:
                im = im.filter(ImageFilter.DETAIL)
            if "filter_edgeEnhance" in processing:
                im = im.filter(ImageFilter.EDGE_ENHANCE)
            if "filter_edgeEnhanceMore" in processing:
                im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
            if "filter_emboss" in processing:
                im = im.filter(ImageFilter.EMBOSS)
            if "filter_findEdges" in processing:
                im = im.filter(ImageFilter.FIND_EDGES)
            if "filter_sharpen" in processing:
                im = im.filter(ImageFilter.SHARPEN)
            if "filter_smooth" in processing:
                im = im.filter(ImageFilter.SMOOTH)
            if "filter_smoothMore" in processing:
                im = im.filter(ImageFilter.SMOOTH_MORE)
        if library_img == "OpenCV":
            if "gray" in processing:
                im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            elif "HSV" in processing:
                im = cv.cvtColor(im, cv.COLOR_BGR2HSV_FULL)
            elif "getChannel" in processing:
                im = im[:, :, int(img_processing[index+1])]
            elif "filterBlur" in processing:
                im = cv.blur(im, (5, 5))
            elif "filterMedianBlur" in processing:
                im = cv.medianBlur(im, int(img_processing[index+1]))
            elif "filterGaussianBlur" in processing:
                im = cv.GaussianBlur(im, (5, 5), 0)
            elif "filterBilateralFilter" in processing:
                im = cv.bilateralFilter(im, 100, 200, 200)
            elif "thresh" in processing:
                im = cv.threshold(im, 127, 255, 0)[1]
            elif "filterMorphology" in processing:
                cv.morphologyEx(src=im, op=cv.MORPH_CLOSE, kernel=cv.getStructuringElement(
                    shape=cv.MORPH_RECT, ksize=(10, 15)), dst=im)
            elif "canny" in processing:
                im = cv.Canny(im, 100, 200)
            elif "histogramEqualization" in processing:
                im = cv.equalizeHist(im)
            elif "filterKMeans" in processing:
                _, label, center = cv.kmeans(np.float32(im.reshape((-1, 3))),
                                             int(img_processing[index + 1]),
                                             None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                             10, cv.KMEANS_RANDOM_CENTERS)
                im = center[label.flatten()].reshape((im.shape))
            elif "filterHOG" in processing:
                pass
            else:
                try:
                    # Skip number configurations
                    float(processing)
                except:
                    warnings.warn("".join(("Image processing \"", processing, "\" not found")))
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


def img_process(arq, library_img, features: str, inverted=False):
    "Get a path if the image, process and return it as pillow/array Image"
    im = open_image(arq, library_img, inverted=False)
    img_processing = [p for p in features.split("_")[0:-1] if p != ""]
    im = processing(im, library_img, img_processing)
    ims = image2images(im, library_img, img_processing)
    return ims
