import cv2 as cv
import numpy as np

import image_processing


def color_contours(im):
    """Find contours in image"""
    returns = []
    im_hue = np.array(im)
    im = cv.threshold(im, 127, 255, 0)[1]
    # cv.morphologyEx(src=im, op=cv.MORPH_CLOSE, kernel=cv.getStructuringElement(shape=cv.MORPH_RECT,
    #                                                                            ksize=(22, 3)), dst=im)
    contours, _ = cv.findContours(im, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    temp = np.array(im_hue)
    cv.drawContours(temp,contours,-1,(0,255,0),-1)
    cv.imwrite("".join(("morph", ".png")),im)
    cv.imwrite("".join(("Contornos", ".png")),temp)
    if len(contours):
        for contour in contours:
            if cv.contourArea(contour) > 3e3:
                mask = np.zeros(im.shape, dtype="uint8")
                cv.drawContours(mask, [contour], -1, 255, -1)
                temp = np.array(im_hue)
                temp[mask == 0] = 0
                returns.append(np.squeeze(normalize(cv.calcHist([temp], [0], None, [256], [0, 256])[1:])))
    return returns


def normalize(list_):
    """Normalize list or array"""
    x_max = max(list_)
    x_min = min(list_)
    difference = x_max-x_min
    if not difference:
        raise Exception("Extract feature is a string of zeros")
    return [(x-x_min)/(difference) for x in list_]


def get_pattern(im):
    return color_contours(im)
