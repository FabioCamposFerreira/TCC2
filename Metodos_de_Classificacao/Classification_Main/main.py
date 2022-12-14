#!/usr/bin/env python
# encoding: utf-8
"""
main.py
====================================
Author: Fábio Campos Ferreira
Interface to manipule machine learns
"""
import time
from typing import List


import numpy as np

import install_dev
import constants
import others
import result_save
import machine_learn

def mls_optimate(mls):
    """Run code to generate optimization graphic"""
    parallel = False
    for ml in mls:
        methods_todo = ml.methods.keys()
        if "SVM" in methods_todo:
            ml.optimization(method="SVM", svm_kernels=["linear", "poly", "rbf", "sigmoid", "chi2", "inter"],
                            quantity_C=2, first_C=0.1, quantity_gamma=1, first_gamma=0.1, quantity_degree=1,
                            first_degree=1, parallel=parallel)
        if "KNN" in methods_todo:
            ml.optimization(method="KNN", quantity_k=10, first_k=1, last_k=10, parallel=parallel)
        if "MLP" in methods_todo:
            ml.optimization(method="MLP", activation=["sigmoid_sym", "gaussian", "relu", "leakyrelu"],
                            quantity_networks=1, quantity_inside_layers=1, range_layer=300, quantity_alpha=1,
                            first_alpha=1, quantity_beta=1, first_beta=1e-2, parallel=parallel)


def mls_start(mls):
    """Run code to generate results to each ml"""
    for ml in mls:
        ml.run()
        result_save.mls_saves(ml, ml.path_output+"MLS Results.csv")


def mls_labeling_only(mls):
    """Run code to generate results to each ml"""
    for ml in mls:
        ml.labeling_only()
        result_save.mls_saves(ml, ml.path_output+"MLS Results.csv")


def mls_construct(todos: List[str],
                  method_libraries: List[str],
                  img_libraries: List[str],
                  features: List[str],
                  data_base_paths: List[str],
                  methods_parameters: List[dict],
                  methods_selected_s: List[List[str]]):
    """Get user interface answers to make and run mls"""
    mls = []
    for method_library in method_libraries:
        for img_library in img_libraries:
            for feature in features:
                for data_base_path in data_base_paths:
                    for method_parameters in methods_parameters:
                        for methods_selected in methods_selected_s:
                            mls += [machine_learn.MachineLearn(method_library, img_library, feature,
                                                 data_base_path, method_parameters, methods_selected)]

    if "optimate" in todos:
        mls_optimate(mls)
    if "start" in todos:
        mls_start(mls)
    if "labeling_only" in todos:
        mls_labeling_only(mls)


if __name__ == "__main__":
    # User Interface
    start_time = time.time()
    todos = constants.todos(start=True, optimate=False, labeling_only=False)
    method_libraries = constants.methods_libraries(OpenCV=True, scikit_learn=False)
    img_libraries = constants.img_libraries(OpenCV=True)
    features = []
    n_features = np.linspace(60, 1000, num=1, dtype=int)
    for n in n_features:
        features += constants.features(histogramFull_256=[False,
                                                          256,
                                                          constants.img_processing(filterGaussianBlur=[1, ""],
                                                                                   filterKMeans=[2, 10*6], HSV=[3, ""], getChannel=[4, 0])],  # [Run?, features len, img_processing**] # [order, option]
                                       histogramFilter_256=[False,
                                                            256,
                                                            constants.img_processing(HSV=[1, ""], getChannel=[2, ""],
                                                                                     filterGaussianBlur=[3, ""])],
                                       histogramReduce_XXX=[False,
                                                            n,
                                                            constants.img_processing(HSV=[1, ""], getChannel=[2, 0],
                                                                                     filterGaussianBlur=[3, ""])],
                                       imagePatches_XXX=[False,
                                                         25*25,
                                                         constants.img_processing(gray=[1, ""], filterGaussianBlur=[2, ""])],
                                       colorContours_255=[False,
                                                          255,
                                                          constants.img_processing(
                                                              gray=[1, ""],
                                                              histogramEqualization=[2, ""],
                                                              filterMedianBlur=[3, 15],
                                                              canny=[4, ""],
                                                              filterMorphology=[5, ""]),
                                                          "processingBreak",
                                                          constants.img_processing(HSV=[1, ""], getChannel=[2, 0], filterGaussianBlur=[3, ""])],
                                       siftHistogram_XXX=[True, n,
                                                          constants.img_processing(
                                                              gray=[1, 0],patchSlip=[1,10])],
                                       gradienteHistogram_XXX=[False, 36*4, ""])
    data_base_paths = constants.data_base_paths(Data_Base_Cedulas=True, temp=False)
    methods_parameters = constants.methods_parameters(
        knn_k=3, mlp_layers=[10],
        svm_c=1, svm_kernel=constants.svm_kernel(inter=True),
        svm_gamma=1, svm_degree=1, activation="sigmoid_sym", alpha=100, beta=100)
    methods_selected = constants.methods_selected(SVM=True, KNN=True, MLP=True)
    mls_construct(todos, method_libraries, img_libraries, features,
                  data_base_paths, methods_parameters, methods_selected)
    print("".join(("Tempo de execução:", str(others.TimeConversion(time.time()-start_time)))))
