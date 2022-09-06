# Author: Fábio Campos Ferreira
# Contain modules for image processing
# In general, the modules open the images from the database, process the images, the result is sent to be used by the modules of feature_extraction.py
from PIL import Image


def open_image(arq):
    """Get a path if the image and return it as pillow Image"""
    im = Image.open(arq).convert(mode='HSV', palette=0)
    # rotate to lay down the image
    l, h = im.size
    if l < h:
        im = im.rotate(angle=90, resample=0, expand=True)
    return im.resize((480,360), resample=Image.BICUBIC)
