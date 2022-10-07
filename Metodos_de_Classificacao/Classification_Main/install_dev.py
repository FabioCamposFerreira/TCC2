import os


def packages_install():
    """Install packages to developers"""
    print("\033[90m")
    os.system("pip install -r requirements.txt")
    print("\033[00m")


def packages_linux():
    """To use OpenCV imshow need install this packages"""
    os.system("sudo apt update")
    os.system("sudo apt install libgtk2.0-dev")
    os.system("sudo apt install pkg-config")


try:
    import numpy
    import PIL
    import sklearn
    import cv2
    import matplotlib
    import joblib
    import scipy
    import bokeh

except:
    packages_install()
