# Author: Fábio Campos Ferreira
# Contains modules to train the classifiers using the features generated by **feature_extraction.py**
# In general, the modules receive the features to train the classifiers, these are saved in binary files with the same name as the classifier.

from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump
import numpy as np
import cv2 as cv


def train(X, y, method_name, method, library, xml_name):
    """Get features and class to train and save method"""
    if library == "OpenCV":
        X = np.matrix(X, dtype=np.float32)
        y = np.array(y)
        if method_name == "MLP":
            y = OneHotEncoder(sparse=False, dtype=np.float32).fit_transform(y.reshape(-1, 1))
        method.train(X, cv.ml.ROW_SAMPLE, y)
        method.save(xml_name.replace("XXX", method_name))
    elif library == "scikit-learn":
        method.fit(X, y)
        dump(method, xml_name.replace("XXX", method_name).replace(".xml", ".joblib"))


def MLP_create(library, mlp_layers, max_iter=300, alpha=2.5):
    """Create and return an OpenCV MLP classifier with the given options"""
    if library == "OpenCV":
        mlp = cv.ml.ANN_MLP_create()
        mlp.setLayerSizes(np.array(mlp_layers))
        mlp.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, alpha, 1.0)
        mlp.setTrainMethod(cv.ml.ANN_MLP_BACKPROP)
        mlp.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, max_iter, 0.01))
        return mlp
    elif library == "scikit-learn":
        mlp = MLPClassifier(hidden_layer_sizes=mlp_layers[1:-1], activation="logistic", max_iter=max_iter)
        return mlp


def KNN_create(library: str, k: int):
    """Create and return an OpenCV KNN classifier with the given options"""
    if library == "OpenCV":
        knn = cv.ml.KNearest_create()
        knn.setDefaultK(k)
        return knn
    elif library == "scikit-learn":
        knn = KNeighborsClassifier(n_neighbors=k)
        return knn


def SVM_create(library, C=1):
    """Create and return an OpenCV SVM classifier with the given options"""
    if library == "OpenCV":
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setC(C)
        svm.setKernel(cv.ml.SVM_LINEAR)
        return svm
    elif library == "scikit-learn":
        svm = SVC(C=C, kernel='linear')
        return svm
