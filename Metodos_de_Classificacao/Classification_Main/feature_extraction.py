# Author: Fábio Campos Ferreira
# Contains modules for extracting features from the images processed by the **image_processing.py** modules
# In general, the modules receive the images processed by the **feature_extraction.py** modules, extracted and treated features with their identification labels, are the input to the **training.py** modules
# When run as **main**, graphics and tables are generated to compare the features
from typing import List

from PIL import Image
import numpy as np
import cv2 as cv

import constants
import image_processing
import training
import classification


def gradienteHistogram(arq: str, feature: str, library_img: str, inverted: bool, n_features):
    im = image_processing.img_process(arq, library_img, feature.split("processingBreak")[0], inverted)
    im_name = "-".join((arq.split("/")[-1], "Inverted"*inverted))
    returns = []
    if library_img == "OpenCV":
        block_size = (2, 2)
        cell_size = (16, 16)
        nbins = 9
        hog = cv.HOGDescriptor(_winSize=(im.shape[1] // cell_size[1] * cell_size[1],
                                         im.shape[0] // cell_size[0] * cell_size[0]),
                               _blockSize=(block_size[1] * cell_size[1],
                                           block_size[0] * cell_size[0]),
                               _blockStride=(cell_size[1], cell_size[0]),
                               _cellSize=(cell_size[1], cell_size[0]),
                               _nbins=nbins)
        # len(hist) = int((im.shape[0]-cell_size[0])/(cell_size[0]))*int((im.shape[1]-cell_size[1])/(cell_size[1]))*36
        hist = hog.compute(im)
    for index, h in enumerate(np.reshape(hist, (int(hist.shape[0]/n_features), n_features))):
        returns.append(["-".join((im_name, str(index))), h])
    return []


def sift_clustering(data_base_path: str, paths: List[str],
                    n_features: int, feature: str, library_img: str, inverted: bool):
    "Generate KNN  dict of the sith"
    print("Gerando dicionario sith...")
    array = sift_vectors("".join((data_base_path, paths[0])), feature, library_img, inverted)
    for arq in paths[1:]:
        array = np.vstack((array, sift_vectors("".join((data_base_path, arq)), feature, library_img, inverted)))
    BoW = clustering(array, library_img, n_features)
    knn_clustering = training.KNN_create("OpenCV", 1)
    return training.train(BoW, range(len(BoW)), "KNN", knn_clustering, "OpenCV", "", False)


def sift_vectors(arq: str, feature: str, library_img: str, inverted: bool):
    "Return all sift vector descritor to one im"
    sift = cv.SIFT_create()
    ims = image_processing.img_process(arq, library_img, feature, inverted)
    des_list = []
    for im in ims:
        _, des = sift.detectAndCompute(im, None)
        if des != None:
            des_list += [des]
    if des_list != []:
        return np.array(des_list[1:0])


def sift_histogram(arq: str, feature: str, library_img: str, inverted: bool, n_features: int, knn_clustering):
    "Convert sift vector descritor to histogram"
    sift = cv.SIFT_create()
    im = image_processing.img_process(arq, library_img, feature, inverted)
    im_name = "-".join((arq.split("/")[-1], "Inverted"*inverted))
    returns = np.zeros(n_features)
    _, des = sift.detectAndCompute(im, None)
    des_length = len(des)
    for d in des:
        index = np.array(knn_clustering.predict(np.matrix(d, dtype=np.float32))[1], dtype=int)[0, 0]
        returns[index] += 1/des_length
    return [[im_name, normalize(returns)]]


def histogram_reduce(arq: str, feature: str, library_img: str, n_features: int, inverted: bool):
    """Recude 256 histogram features to n_features"""
    im_name = "-".join((arq.split("/")[-1], "Inverted"*inverted))
    hist = histogramFull(arq, feature, library_img, inverted)[0][1]
    step = int(256/n_features)
    new_hist = []
    for index in range(n_features):
        new_hist += [sum(hist[step*index:(step*index)+step])]
    return [[im_name, normalize(new_hist)]]


def histogramFull(arq: str, feature: str, library_img: str, inverted: bool):
    """Receive path image and return histogram of the channel H"""
    im = image_processing.img_process(arq, library_img, feature, inverted)
    im_name = "-".join((arq.split("/")[-1], "Inverted"*inverted))
    if library_img == "Pillow":
        h = im.getchannel(channel=0).histogram(mask=None, extrema=None)
    elif library_img == "OpenCV":
        h = np.squeeze(cv.calcHist([im], [0], None, [256], [0, 256])).tolist()
    return [[im_name, normalize(h)]]


def histogram_filter(im: np.ndarray, im_name: str, library_img: str):
    """Receive image and return histogram of the channel H excruing pixels with low saturation and value in extrems"""
    if library_img == "Pillow":
        im = np.array(im)
        im = np.vstack(im)
    im = im[(im[:, :, 1] > 255 - 255 * constants.SATURATION_TOLERANCE)
            & (im[:, :, 2] > 255 / 2 - 255 / 2 * constants.VALUE_TOLERANCE)
            & (im[:, :, 2] < 255 / 2 + 255 / 2 * constants.VALUE_TOLERANCE)]
    return [[im_name, np.histogram(im[:, 0], bins=range(256+1))[0]]]


def normalize(list_):
    """Normalize list or array"""
    x_max = max(list_)
    x_min = min(list_)
    difference = x_max-x_min
    if not difference:
        raise Exception("Extract feature is a string of zeros")
    return [(x-x_min)/(difference)*100 for x in list_]


def clustering(array: np.ndarray, library_img: str, k=2):
    "clustering list vectors to k vectors"
    if library_img == "OpenCV":
        _, _, center = cv.kmeans(array, k,
                                 None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                 10, cv.KMEANS_RANDOM_CENTERS)
    return center


def get_features(arq: str, feature, library_img, n_features=10, inverted=False, knn_clustering=None):
    """Extract image features
    """
    if "histogramFull" in feature:
        features = histogramFull(arq, feature, library_img, inverted)
    elif "histogramFilter" in feature:
        features = histogram_filter(arq, feature, library_img, inverted)
    elif "histogramReduce" in feature:
        features = histogram_reduce(arq, feature, library_img, n_features, inverted)
    elif "siftHistogram" in feature:
        features = sift_histogram(arq, feature, library_img, inverted, n_features, knn_clustering)
    elif "gradienteHistogram" in feature:
        features = gradienteHistogram(arq, feature, library_img, inverted, n_features)
    else:
        Exception("Feature not found!")
    return features
