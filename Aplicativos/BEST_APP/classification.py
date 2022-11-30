# Author: Fábio Campos Ferreira
#

import csv

import numpy as np
import cv2 as cv
from kivy.utils import platform

import feature_extraction


def classify(im):
    """Extract image pattern

    Args:
        im : 
            pillow Image

    Returns:
        y : int or None
            The class label or None
    """
    classifier = "SVM"
    pattern = feature_extraction.get_pattern(im)
    if platform == "linux":
        with open("".join(("Pattern", ".csv")), "w") as f:
            writer = csv.writer(f)
            writer.writerow("Características")
            writer.writerow(pattern)
    # if platform == "android":
    #     with open("".join(("/storage/emulated/0/Download/Pattern", ".csv")), "w") as f:
    #         writer = csv.writer(f)
    #         writer.writerow("Características")
    #         writer.writerow(pattern)
    if len(pattern) > 0:
        pattern = np.matrix(pattern, dtype=np.float32)
        clsf = read_object(classifier+".xml")
        result = np.squeeze(clsf.predict(pattern)[1])
        if result.shape != ():
            y = np.bincount(list(result)).argmax()
            if len(result == y) > len(result)/2:
                return y
        else:
            return result


def read_object(arq):
    """Le objeto no arquivo

    Args:
        arq: string
            Localização do arquivo

    Returns:
        : ??
            O objeto contido no arquivo
    """
    return cv.ml_SVM.load(arq)
