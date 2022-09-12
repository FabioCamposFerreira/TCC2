# Contains modules for classifying a new image using classifiers trained by **training.py**
# In general, the modules receive the features to classify them, returning the results.

import numpy as np
import cv2 as cv


def labeling(X, method_name,  library):
    """Receives feature, classifier name, library to use  and return list class predictions"""
    if library == "OpenCV":
        method = cv.ml_SVM.load(method_name+".xml")
        X = np.matrix(X, dtype=np.float32)
        class_predictions = np.array(method.predict(X)[1], dtype=np.int)
        return class_predictions
    elif library == "scikit-learn":
        pass
