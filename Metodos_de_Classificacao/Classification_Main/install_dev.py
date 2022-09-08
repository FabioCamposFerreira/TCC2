import os


def packages_install():
    """Install packages to developers"""
    print("\033[90m")
    os.system("pip install -r requirements.txt")
    print("\033[00m")


try:
    import numpy
    import PIL
    import sklearn
    import cv2
except:
    packages_install()
