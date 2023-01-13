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
            ml.optimization(method="SVM", svm_kernels=["linear", "poly",  "rbf", "sigmoid", "chi2", "inter"],
                            quantity_C=1, first_C=0.1, quantity_gamma=5, first_gamma=0.1, quantity_degree=1,
                            first_degree=1, parallel=parallel)
        if "KNN" in methods_todo:
            ml.optimization(method="KNN", quantity_k=2, first_k=1, last_k=10, parallel=parallel)
        if "MLP" in methods_todo:
            ml.optimization(method="MLP", activation=["sigmoid_sym"],  # , "gaussian", "relu"],
                            quantity_networks=1, quantity_inside_layers=1, range_layer=300, quantity_alpha=5,
                            first_alpha=.001, last_alpha=2.5, quantity_beta=1, first_beta=1, parallel=parallel)


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
    # [Run?, features len, img_processing**] # [order, option]
    features += constants.features(histogramFull_XXX=[False, 256,
                                                      constants.img_processing(HSV=[1, ""], getChannel=[2, 0])])
    n_features = [5]  # grid search kernel size blur
    for n in n_features:
        features += constants.features(histogramFull_XXX=[False, 256, constants.img_processing(filterGaussianBlur=[1, n], HSV=[
                                       2, ""], getChannel=[3, 0])])  # [Run?, features len, img_processing**] # [order, option]
        features += constants.features(histogramFull_XXX=[False, 256, constants.img_processing(filterMedianBlur=[1, n], HSV=[
                                       2, ""], getChannel=[3, 0])])  # [Run?, features len, img_processing**] # [order, option]
    n_features = [5]  # grid search kernel size blur to bilateral
    for n in n_features:
        features += constants.features(histogramFull_XXX=[False, 256, constants.img_processing(filterBilateral=[1, n], HSV=[
                                       2, ""], getChannel=[3, 0])])  # [Run?, features len, img_processing**] # [order, option]
    n_features = [200]  # grid search k from k means
    for n in n_features:
        features += constants.features(
            histogramFull_XXX=[False, 256, constants.img_processing(
                filterKMeans=[1, n],
                HSV=[2, ""],
                getChannel=[3, 0])])  # [Run?, features len, img_processing**] # [order, option]
    features += constants.features(histogramFull_XXX=[False, 255, constants.img_processing(
        HSV=[1, ""],
        getChannel=[2, 0],
        histogramEqualization=[3, ""])])  # [Run?, features len, img_processing**] # [order, option]
    pixels_len = constants.RESOLUTION[0]*constants.RESOLUTION[1]
    n_features = np.linspace(int(pixels_len*0.1), pixels_len, 4, dtype=int)  # grid search best size paths
    for n in n_features:
        features += constants.features(histogramFull_XXX=[False, 256, constants.img_processing(
            HSV=[1, ""],
            getChannel=[2, 0],
            patchSlip=[3, n])])  # [Run?, features len, img_processing**] # [order, option]
    features += constants.features(histogramFull_XXX=[False, 255, constants.img_processing(
        gray=[1, 0],
        histogramEqualization=[2, ""],
        filterMedianBlur=[3, 15],
        canny=[4, ""],
        filterMorphology=[5, ""],
        processingBreak=[6, ""],
        HSV=[7, ""],
        getChannel=[8, 0],
        contourSlip=[9, ""])])  # [Run?, features len, img_processing**] # [order, option]
    #Begin teste
    features += constants.features(histogramFull_XXX=[True, 255, constants.img_processing(
        gray=[1, 0],
        histogramEqualization=[2, ""],
        filterMedianBlur=[3, 15],
        canny=[4, ""],
        filterMorphology=[5, ""],
        processingBreak=[6, ""],
        filterGaussianBlur=[7, 1],
        contourSlip=[9, ""])])  
    # END teste
    features += constants.features(histogramFull_XXX=[False, 255, constants.img_processing(
        filterKMeans=[1, 2],
        gray=[2, 0],
        thresh=[3, 0],
        processingBreak=[4, ""])+"_" +
        constants.img_processing(
        HSV=[5, ""],
        getChannel=[6, 0],
        segmentation=[7, ""])])  # [Run?, features len, img_processing**] # [order, option]
    features += constants.features(histogramFilter_256=[False,
                                                        256,
                                                        constants.img_processing(HSV=[1, ""])])
    n_features = [60]  # np.linspace(6, 255, 4, dtype=int)  # grid search best reduction
    for n in n_features:
        features += constants.features(histogramReduce_XXX=[False,
                                                            n,
                                                            constants.img_processing(HSV=[1, ""], getChannel=[2, 0],)])
    # np.linspace(int(pixels_len*0.1), pixels_len, 4, dtype=int)  # grid search best size paths
    n_features = [pixels_len]
    for n in n_features:
        features += constants.features(imageFull_XXX=[False, n,
                                                         constants.img_processing(HSV=[1, ""], getChannel=[2, 0])])
    n_features = [600]  # np.linspace(6, 600, 4, dtype=int)  # grid search best n sift
    for n in n_features:
        features += constants.features(siftHistogram_XXX=[False, n,
                                                        constants.img_processing(
                                                            gray=[1, 0], dontSlip=[2, 0])])
        features += constants.features(siftHistogram_XXX=[False, n,
                                                        constants.img_processing(
                                                            gray=[1, 0], filterGaussianBlur=[2, 5])])
        features += constants.features(siftHistogram_XXX=[False, n,
                                                        constants.img_processing(
                                                            gray=[1, 0], filterMedianBlur=[2, 5])])
        features += constants.features(siftHistogram_XXX=[False, n,
                                                        constants.img_processing(
                                                            gray=[1, 0], filterBilateral=[2, 5])])
        # HOG
        n_features = int(pixels_len*0.1)
        features += constants.features(gradienteHistogram_XXX=[False, 54288,
                                                        constants.img_processing(
                                                            gray=[1, 0], patchSlip=[0,n_features])])
        features += constants.features(gradienteHistogram_XXX=[False, 54288,
                                                        constants.img_processing(
                                                            gray=[1, 0], patchSlip=[0,n_features], filterGaussianBlur=[3, 5])])
        features += constants.features(gradienteHistogram_XXX=[False, 54288,
                                                        constants.img_processing(
                                                            gray=[1, 0], patchSlip=[0,n_features], filterMedianBlur=[3, 5])])
        features += constants.features(gradienteHistogram_XXX=[False, 54288,
                                                        constants.img_processing(
                                                            gray=[1, 0], patchSlip=[0,n_features], filterBilateral=[3, 5])])

    data_base_paths=constants.data_base_paths(Data_Base_Cedulas=False, temp=False, Data_Base_Refencia=True)
    methods_parameters=constants.methods_parameters(
        knn_k=3, mlp_layers=[10],
        svm_c=1, svm_kernel=constants.svm_kernel(inter=True),
        svm_gamma=1, svm_degree=1, activation="sigmoid_sym", alpha=0.5, beta=1)
    methods_selected=constants.methods_selected(SVM=True, KNN=False, MLP=False)
    mls_construct(todos, method_libraries, img_libraries, features,
                  data_base_paths, methods_parameters, methods_selected)
    print("".join(("Tempo de execução:", str(others.TimeConversion(time.time()-start_time)))))
