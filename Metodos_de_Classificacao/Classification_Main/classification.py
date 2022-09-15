﻿# Contains modules for classifying a new image using classifiers trained by **training.py**
# In general, the modules receive the features to classify them, returning the results.

from joblib import load
import numpy as np
import cv2 as cv
import training

def labeling(X, method_name,  library, xml_name):
    """Receives feature, classifier name, library to use  and return list class predictions"""
    if library == "OpenCV":
        if method_name == "SVM":
            method = cv.ml_SVM.load(xml_name.replace("XXX",method_name))
        if method_name == "MLP":
            method = cv.ml_ANN_MLP.load(xml_name.replace("XXX",method_name))
        elif method_name == "KNN":
            fs = cv.FileStorage(xml_name.replace("XXX",method_name), cv.FILE_STORAGE_READ)
            knn_xml = fs.getNode('opencv_ml_knn')
            default_k = int(knn_xml.getNode('default_k').real())
            samples = knn_xml.getNode('samples').mat()
            responses = knn_xml.getNode('responses').mat()
            fs.release
            method = training.KNN_create(library, default_k)
            method.train(samples, cv.ml.ROW_SAMPLE, responses)
        X = np.matrix(X, dtype=np.float32)
        class_predictions = np.array(method.predict(X)[1], dtype=np.int)
        return class_predictions
    elif library == "scikit-learn":
        method = load(xml_name.replace("XXX", method_name).replace(".xml", ".joblib")) 
        class_predictions = np.array(method.predict(X), dtype=np.int)
        return class_predictions