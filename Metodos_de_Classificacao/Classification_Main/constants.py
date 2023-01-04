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


def img_processing(
        HSV=[False, 0],
        getChannel=[False, 0],
        filterBlur=[False, 0],
        filterGaussianBlur=[False, 0],
        filterBilateral=[False, 0],
        filterMorphology=[False, 0],
        gray=[False, 0],
        histogramEqualization=[False, 0],
        filterMedianBlur=[False, 0],
        canny=[False, 0],
        filterKMeans=[False, 2],
        filerHOG=[False, 0],
        increseContrastBrightness=[False, 0],
        dontSlip=[False, 0],
        patchSlip=[False, 0],
        contourSlip=[False, 0]):
    l = [tuple_ for tuple_ in locals().items() if type(tuple_[1]) == list]
    l = {k: v for k, v in sorted(l, key=lambda item: item[1])}
    return "_".join((["_".join(filter(None, (key, str(l[key][1])))) for key in l if l[key][0] != False]))


def features(histogramFull_256=[0, None, None],  # [True/False,n of the features,image processing]
             histogramFilter_256=[0, None, None],
             imagePatches_XXX=[0, 0, None],
             imageContours_XXX=[0, 0, None],
             histogramReduce_XXX=[0, 0, None],
             #  colorContours_255=[0, None, None],
             siftHistogram_XXX=[0, None, None],
             gradienteHistogram_XXX=[0, None, None]):
    l = dict(locals())
    return ["_".join(("_".join((l[key][2:])), key.replace("XXX", str(l[key][1])))) for key in l if l[key][0] == 1]


def data_base_paths(temp=0, Data_Base_Cedulas=0, Data_Base_Refencia=0):
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
AREA_MIN = 3e3
knn_clustering = []
COLORS=["#FF0000", "#0000FF", "#FFA500", "#800080", "#ABAB13", "#248524", "#00008B", "#A52A2A", "#000000",
                 "#800000", "#008000", "#FF00FF", "#808000", "#FFC0CB", "#7FFFD4", "#00FFFF", "#ADD8E6"]
MARKERS = ["circle", "diamond", "triangle", "square", "plus", "star",
           "triangle_pin", "hex", "inverted_triangle", "asterisk", "cross", "x", "y"]
