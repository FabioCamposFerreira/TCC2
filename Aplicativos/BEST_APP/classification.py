# Author: Fábio Campos Ferreira
#

import image_processing
import numpy as np
import cv2 as cv


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
    pattern = get_pattern(im)
    pattern = np.matrix(pattern, dtype=np.float32)
    clsf = read_object(classifier+".xml")
    y = clsf.predict(pattern)[1][0][0]
    # id_max = predct_proba[0].argmax()
    # analyze criteria for sure
    # if predct_proba[0][id_max] > .5:
    #     y = id_max
    # else:
    #     y = None
    return y


def get_pattern(im):
    """Extract image pattern

    Args:
        im: 
            pillow Image

    Returns:
        : list
            The patterns
    """
    return image_processing.histogram(im)


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
