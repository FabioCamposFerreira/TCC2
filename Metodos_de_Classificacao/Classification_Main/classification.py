# Contains modules for classifying a new image using classifiers trained by **training.py**
# In general, the modules receive the features to classify them, returning the results.

from sklearn.preprocessing import OneHotEncoder
from joblib import load
import numpy as np
import cv2 as cv
import training


def labeling(X: list, y_full: list, method_name: str,  library: str, xml_name: str, method) -> int:
    """Receives feature, classifier name, library to use  and return list class predictions"""
    if library == "OpenCV":
        if method == []:
            if method_name == "SVM":
                method = cv.ml_SVM.load(xml_name.replace("XXX", method_name))
            if method_name == "MLP":
                method = cv.ml_ANN_MLP.load(xml_name.replace("XXX", method_name))
            elif method_name == "KNN":
                fs = cv.FileStorage(xml_name.replace("XXX", method_name), cv.FILE_STORAGE_READ)
                knn_xml = fs.getNode('opencv_ml_knn')
                default_k = int(knn_xml.getNode('default_k').real())
                samples = knn_xml.getNode('samples').mat()
                responses = knn_xml.getNode('responses').mat()
                fs.release
                method = training.KNN_create(library, default_k)
                method.train(samples, cv.ml.ROW_SAMPLE, responses)
        X = np.matrix(X, dtype=np.float32)
        y_predict = np.array(method.predict(X)[1], dtype=int)
        if method_name == "MLP":
            y_mlp=""
            enc = OneHotEncoder(sparse=False, dtype=np.float32, handle_unknown="ignore")
            _ = enc.fit_transform(np.array(y_full).reshape(-1, 1))
            max_y=max(y_predict[0])
            doubt = np.where(max_y==y_predict[0])
            for index in  doubt[0]:
                y_temp = y_predict
                y_temp[0,doubt]=0
                y_temp[0,index]=max_y
                y_temp = enc.inverse_transform(y_temp)
                y_temp[y_temp == None] = 0
                y_mlp="9".join((y_mlp,str(y_temp[0,0])))
            y_predict = y_mlp
        return y_predict
    elif library == "scikit-learn":
        method = load(xml_name.replace("XXX", method_name).replace(".xml", ".joblib"))
        y_predict = np.array(method.predict([X]), dtype=np.int)
        return y_predict
