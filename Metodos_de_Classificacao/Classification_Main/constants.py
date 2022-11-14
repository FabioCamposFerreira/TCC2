#!/usr/bin/env python
# encoding: utf-8
"""
Contain constants and options to use, The functions with return the select parameters
"""
from typing import List


def todos(start=0,  optimate=0,  labeling_only=0):
    l = locals()
    return [key for key in l if l[key] == 1]


def methods_libraries(OpenCV=0, scikit_learn=0):
    l = locals()
    return [key.replace("_", "-") for key in l if l[key] == 1]


def img_libraries(OpenCV=0, Pillow=0):
    l = locals()
    return [key for key in l if l[key] == 1]


def img_processing(HSV=0, get_H=0, filter_blur=0, filter_gaussian_blur=0,filter_morphology=0):
    l = locals()
    return [[key for key in l if l[key] == 1]]


def features(histogram_256=[0, None],
             histogram_filter_256=[0, None],
             image_patches_XXX=[0, 0],
             image_contours_XXX=[0, 0],
             histogram_reduce_XXX=[0, 0],
             color_contours_255=[0,None]):
    l = locals()
    return [key.replace("XXX", str(l[key][1])) for key in l if l[key][0] == 1]


def data_base_paths(temp=0, Data_Base_Cedulas=0):
    l = locals()
    return ["".join(("../../Data_Base/", key, "/")) for key in l if l[key] == 1]


def data_base_paths(temp=0, Data_Base_Cedulas=0):
    l = locals()
    return ["".join(("../../Data_Base/", key, "/")) for key in l if l[key] == 1]


def svm_kernel(linear=0, poly=0, rbf=0, sigmoid=0, chi2=0, inter=0) -> str:
    l = locals()
    return max(l, key=l.get)


def methods_parameters(knn_k: int, mlp_layers: List[int], svm_kernel: svm_kernel, svm_c: float, svm_gamma: float,
                       svm_degree: float, activation: str, alpha=float, beta=float):

    return [{"knn_k": knn_k, "mlp_layers": mlp_layers, "svm_kernel": svm_kernel, "svm_c": svm_c, "svm_gamma": svm_gamma,
            "svm_degree": svm_degree, "activation": activation, "alpha": alpha, "beta": beta}]


def methods_selected(SVM=0, KNN=0, MLP=0):
    l = locals()
    return [[key for key in l if l[key] == 1]]


RESOLUTION = (854, 480)
SATURATION_TOLERANCE = 0.8
VALUE_TOLERANCE = 0.9
