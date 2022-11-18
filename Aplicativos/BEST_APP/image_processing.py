import cv2 as cv
import numpy as np
from kivy.utils import platform

import constants


def processing(im: np.ndarray):
    """Make processing image"""
    im = cv.threshold(im, 127, 255, 0)[1]
    return im


def process_image(im: np.array):

    im = cv.GaussianBlur(im, (5, 5), 0)
    # rotate to lay down the image
    h, l = im.shape[:2]
    if l < h:
        im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
    im = cv.resize(im, constants.RESOLUTION, cv.INTER_NEAREST)
    if platform == "android":
        cv.imwrite("".join(("/storage/emulated/0/Download/Imagem Processada", ".png")), im)
    cv.imwrite("".join(("Imagem Processada", ".png")), im)
    im = cv.cvtColor(im, cv.COLOR_BGR2HSV_FULL)
    im = im[:, :, 0]
    return im


def process_texture(texture):

    im = cv.cvtColor(np.frombuffer(texture.pixels, dtype=np.uint8).reshape(texture.height,texture.width,4),cv.COLOR_RGBA2BGR)
    if platform == "android":
        cv.imwrite("".join(("/storage/emulated/0/Download/Imagem Capturada", ".png")), im)
    return process_image(im)
