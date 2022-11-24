#!/usr/bin/env python
# encoding: utf-8
"""
main.py
====================================
Author: Fábio Campos Ferreira
Contains step by step instructions for performing image processing, feature extraction, feature training and classification of unknown images
Several configuration options are presented at each step for later comparison
"""
import time
import os
import random
import time
from multiprocessing import Manager, Process
from typing import List

import cv2 as cv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, mean_squared_error
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error, precision_score, recall_score)

import install_dev
import constants
import feature_extraction
import training
import classification
import others
import result_save


class MachineLearn:
    def __init__(self, method_library: str, library_img: str,
                 feature: str, data_base_path: str, method_parameters: dict, methods_selected: List[str]):
        self.data_base = os.listdir(data_base_path)
        self.data_base.sort(key=others.images_sort)
        feature_temp = feature.split("_")
        self.layer_first = int(feature_temp[-1])
        self.parameters = {
            # Global
            "data_base_path": data_base_path,
            "method_library": method_library,
            "library_img": library_img,
            "feature": "_".join(feature_temp[:-1]),
        }
        # Results
        self.images_features = []  # [[name,feature],[name,feature],...]
        self.y = []
        self.X = []
        self.results = []
        self.accuracy = {}
        self.precision = {}
        self.confusion_matrix = {}
        self.recall = {}
        self.meansquare_error = {}
        # Files with results
        self.path_output = "./Output/"
        self.path_classifiers = self.path_output+"Classifiers/"
        self.path_features = self.path_output+"Patterns/"
        self.path_graphics = (self.path_output
                              + "Graphics/"
                              + "XXX,"
                              + ",".join(
                                  ["data_base_path"+"="+"".join(self.parameters["data_base_path"].split("/")[-2:]),
                                   "library_img"+"="+self.parameters["library_img"],
                                   "feature"+"="+self.parameters["feature"]]))
        self.path_results = self.path_output+"Results/"
        self.files_name = (
            "XXX-"
            + ",".join(p + "=" + str(self.parameters[p]).split("/")[-2:][0] for p in self.parameters))
        self.csv_results = self.path_results+self.files_name.replace("XXX", "Resultados")+".csv"
        self.csv_features = self.path_features+self.files_name.replace("XXX", "Características")+".csv"
        self.xml_name = self.path_classifiers+self.files_name+".xml"
        # Construct classifiers
        self.methods = {}
        if "SVM" in methods_selected:
            self.methods["SVM"] = training.SVM_create(library=self.parameters["method_library"],
                                                      C=method_parameters["svm_c"],
                                                      kernel=method_parameters["svm_kernel"],
                                                      gamma=method_parameters["svm_gamma"],
                                                      degree=method_parameters["svm_degree"])
        if "KNN" in methods_selected:
            self.methods["KNN"] = training.KNN_create(library=self.parameters["method_library"],
                                                      k=method_parameters["knn_k"])
        if "MLP" in methods_selected:
            self.mlp_layers = ([self.layer_first] + method_parameters["mlp_layers"] +
                               [len(list(dict.fromkeys([arq.split(".")[0] for arq in self.data_base])))])
            self.methods["MLP"] = training.MLP_create(
                library=self.parameters["method_library"],
                mlp_layers=self.mlp_layers,
                activation=method_parameters["activation"],
                alpha=method_parameters["alpha"],
                beta=method_parameters["beta"])

    def show(self):
        """Show the classifications parameters"""
        print("\nParâmetros usados: ")
        for parameter in self.files_name.replace("./results/XXX-", "").split(","):
            print("\t\033[91m {}\033[00m".format(parameter))
        print("CSV com as características: " + "\033[91m {}\033[00m".format(self.csv_features))
        print("CSV com os resultados: " + "\033[91m {}\033[00m".format(self.csv_results))

    def setup_feature(self):
        """Do feature extraction"""
        try:
            self.images_features = result_save.features_open(self.csv_features)
        except:
            self.images_features = []
            print("Extraindo as características")
            actual = 0
            progress_bar = others.ProgressBar("Extraindo características", len(self.data_base), 0)
            knn_clustering = None
            if "siftHistogram" in self.parameters["feature"]:
                knn_clustering = feature_extraction.sift_clustering(self.parameters["data_base_path"],
                                                   self.data_base, self.layer_first,
                                                   self.parameters["feature"],
                                                   self.parameters["library_img"], False)
            for arq in self.data_base:
                actual += 1
                progress_bar.print(actual)
                path = "".join((self.parameters["data_base_path"], arq))
                self.images_features += feature_extraction.get_features(path,
                                                                        self.parameters["feature"],
                                                                        self.parameters["library_img"],
                                                                        self.layer_first,
                                                                        knn_clustering=knn_clustering)
                self.images_features += feature_extraction.get_features(path,
                                                                        self.parameters["feature"],
                                                                        self.parameters["library_img"],
                                                                        self.layer_first,
                                                                        inverted=True,
                                                                        knn_clustering=knn_clustering)
            progress_bar.end()
            result_save.features_save(self.csv_features, self.images_features)
            print("Salvando gráficos em "+self.path_graphics)
            result_save.graphics_save(self.path_graphics, self.images_features)
        self.y = [int(row[0].split(".")[0]) for row in self.images_features]
        self.X = [row[1] for row in self.images_features]

    def setup_train(self, X: List[float], y: List[int], file_save: True):
        """Do training and save classifiers in files"""
        classifier = {}
        for method in self.methods:
            classifier[method] = training.train(X, y, method, self.methods[method],
                                                self.parameters["method_library"],
                                                self.xml_name, file_save)
        return classifier

    def labeling(self, X: List[float], y_correct: int, y_full: list, img_name: str, classifier: dict = {}):
        """Do labeling and update results"""
        if classifier == {}:
            classifier = dict.fromkeys(self.methods.keys(), [])
        result = [img_name, y_correct]
        for method in self.methods:
            start_time = time.time()
            y_predict = classification.labeling(X, y_full, method, self.parameters["method_library"],
                                                self.xml_name, classifier[method])
            result += [int(y_predict), time.time()-start_time]
        return result

    def setup_metrics(self):
        """Generate metrics of the result classification"""
        results = np.array(self.results)
        classes_correct = results[:, 1]
        classes_correct = np.array(classes_correct, dtype=int)
        classes_set = list(set(classes_correct))
        results = results[:, 2:]  # remove image name and class correct
        results = np.array(results, dtype=float)
        results = np.array(results, dtype=int)
        for index, method in enumerate(self.methods):
            if method == "MLP":
                results[:, (2*index)] = [int(str(result).split("9")[1]) for result in results[:, (2*index)]]
            self.accuracy[method] = accuracy_score(classes_correct, results[:, (2*index)])
            self.precision[method] = np.average(precision_score(classes_correct,
                                                                results[:, (2 * index)],
                                                                average=None, zero_division=0),
                                                weights=classes_set)
            self.confusion_matrix[method] = confusion_matrix(classes_correct, results[:, (2*index)])
            self.recall[method] = np.average(
                recall_score(classes_correct, results[:, (2 * index)], average=None,
                             zero_division=0),
                weights=classes_set)
            self.meansquare_error[method] = mean_squared_error(classes_correct, results[:, (2 * index)])

    def setup_save(self):
        """Save the results fo the labeling"""
        print("Salvando Resultados em "+self.csv_results)
        result_save.save(self.csv_results, self.methods,  np.array(self.results))

    def validation(self, X: List[List[float]], y: List[List[int]], index: int,
                   results: list = None, classifier_save: bool = False):
        """Train and classify data to one validation in cross validation"""
        # To Remove inverted feature: jump next feature or previous above features
        im_name = self.images_features[index][0].split("-")[0]
        index_pause = 0
        index_restart = 0
        length = len(self.images_features)-1
        for index_test in range(index, -1, -1):
            if self.images_features[index_test][0].split("-")[0] != im_name:
                index_pause = index_test
                break

        for index_test in range(index, length):
            if self.images_features[index_test][0].split("-")[0] != im_name:
                index_restart = index_test
                break
        X_train = X[:index_pause]+X[index_restart:]
        y_train = y[:index_pause]+y[index_restart:]
        # Load classifier
        classifier = self.setup_train(X_train, y_train, file_save=classifier_save)
        # Labeling
        if results == None:
            return self.labeling(X[index], y[index], y, self.images_features[index][0], classifier=classifier)
        else:
            results.append(self.labeling(X[index], y[index], y, self.images_features[index][0], classifier=classifier))

    def validation_parallel(self,  X: List[List[float]], y: List[int], features_len: int):
        """Run multiples self.validation in parallel"""
        progress_bar = others.ProgressBar("Fazendo validação cruzada", features_len, 0)
        processes = []
        results = Manager().list()
        for index in range(features_len):
            p = Process(target=self.validation, args=(X, y, index, results, False))
            processes.append(p)
            p.start()
        for index, process in enumerate(processes):  # waiting finish all process
            progress_bar.print(index)
            process.join()
        progress_bar.end()
        self.results = list(results)

    def validation_serial(self,  X: List[List[float]], y: List[int], features_len: int):
        """Run multiples self.validation in serial"""
        progress_bar = others.ProgressBar("Fazendo validação cruzada", features_len, 0)
        results = []
        for index in range(features_len):
            progress_bar.print(index)
            results.append(self.validation(X, y, index, classifier_save=True))
        self.results = list(results)

    def cross_validation(self, X, y, features_len, parallel: bool = True):
        """Make cross validation one-leave-out to each method"""
        if parallel == True:
            self.validation_parallel(X, y, features_len)
        else:
            self.validation_serial(X, y, features_len)

    def cross_validation_parallel(self, X, y, features_len, method, parameters, results_xylabel):
        """Call self.cross validation to run in parallel form"""
        time_start = time.time()
        self.cross_validation(X, y, features_len, False)
        results_xylabel['tempo (s)'] += [time.time() - time_start]
        self.setup_metrics()
        results_xylabel['accuracy'] += [self.accuracy[method]]
        for key in parameters.keys():
            results_xylabel[key] += [parameters[key]]

    def parameters_combination(self, keys: List[str], grid: dict, parameters: dict, method: str,
                               progress_bar: others.ProgressBar, actual: int, results_xylabel: list,
                               process_parallel: Process = None) -> int:
        """Recursive function that make cross validation to each combination parameters in grid"""
        try:
            for parameter in grid[keys[0]]:
                parameters[keys[0]] = parameter
                actual = self.parameters_combination(keys[1:], grid, parameters, method, progress_bar, actual,
                                                     results_xylabel, process_parallel)
        except IndexError:
            if process_parallel == None:
                progress_bar.print(actual)
            if method == "SVM":
                self.methods = {"SVM": training.SVM_create(self.parameters["method_library"],
                                                           C=parameters["C"],
                                                           kernel=parameters["kernel"],
                                                           gamma=parameters["gamma"],
                                                           degree=parameters["degree"])}
            elif method == "KNN":
                self.methods = {"KNN": training.KNN_create(self.parameters["method_library"], k=parameters["k"])}
            elif method == "MLP":
                self.methods = {"MLP": training.MLP_create(library=self.parameters["method_library"],
                                                           mlp_layers=parameters["layers"],
                                                           activation=parameters["activation"],
                                                           alpha=parameters["alpha"],
                                                           beta=parameters["beta"])}
            if process_parallel != None:
                p = Process(target=self.cross_validation_parallel, args=(self.X, self.y, len(self.images_features),
                                                                         method, parameters, results_xylabel))
                process_parallel.append(p)
                p.start()
            else:
                self.cross_validation_parallel(self.X, self.y, len(self.images_features), method, parameters,
                                               results_xylabel)
            return actual+1
        return actual

    def method_optimization(self, grid: dict, method: str, parallel: bool = True):
        """Optimize method and save results in graphic"""
        manager_dict = {"accuracy": [], "tempo": []}
        for key in grid.keys():
            manager_dict[key] = []
        results_xylabel = Manager().dict(manager_dict)
        print("Otimizando {}: ".format(method)+str(grid).replace("{", "").replace("}", ""))
        total = 1
        for key in grid.keys():
            total *= len(grid[key])
        print("\n")  # to progress bar not bug
        progress_bar = others.ProgressBar("Otimizando "+method, total, 0)
        if parallel == True:
            process_parallel = []
            self.parameters_combination(keys=list(grid.keys()),
                                        grid=grid, parameters={},
                                        method=method, progress_bar=None, actual=0, results_xylabel=results_xylabel,
                                        process_parallel=process_parallel)
            for index, p in enumerate(process_parallel):
                progress_bar.print(index)
                print("")
                p.join()
        else:
            self.parameters_combination(keys=list(grid.keys()),
                                        grid=grid, parameters={},
                                        method=method, progress_bar=progress_bar, actual=0,
                                        results_xylabel=results_xylabel)
        result_save.optimization_graph(dict(results_xylabel),
                                       self.path_graphics.replace("XXX", "".join((method, "_otimização"))))

    def optimization(
            self, method: str, svm_kernels=["linear", "poly", "rbf", "sigmoid", "chi2", "inter"],
            quantity_C=10, first_C=0.1, quantity_gamma=10, first_gamma=0.1, quantity_degree=10, first_degree=1,
            quantity_k=100, first_k=1, last_k=100,
            activation=["identity", "sigmoid_sym", "gaussian", "relu", "leakyrelu"],
            quantity_layers=10, quantity_insidelayers=1, range_layer=10, quantity_alpha=10, first_alpha=1e-6,
            quantity_beta=10, first_beta=1e-2):
        """Optimize classifier selected with your parameters range"""
        self.setup_feature()
        print("Começando processo de otimização...")
        if method == "SVM":
            grid_svc = {"kernel": svm_kernels,
                        "C": np.linspace(first_C, 1000, num=quantity_C, dtype=float),
                        "gamma": np.linspace(first_gamma, 100, num=quantity_gamma, dtype=float),
                        "degree": np.linspace(first_degree, 10, num=quantity_degree, dtype=int)}
            if 1:
                grid_svc = {"kernel": svm_kernels,
                            "C": cv.ml.ParamGrid_create(first_C, 1000, 1),
                            "gamma": cv.ml.ParamGrid_create(first_gamma, 100, 1),
                            "any": cv.ml.ParamGrid_create(1e-3, 100, 1),
                            "degree": cv.ml.ParamGrid_create(first_degree, 10, 1)}
                self.methods["SVM"].trainAuto(
                    np.matrix(self.X, dtype=np.float32),
                    cv.ml.ROW_SAMPLE, np.array(self.y),
                    len(self.X),
                    grid_svc["C"],
                    grid_svc["gamma"],
                    grid_svc["any"],
                    grid_svc["any"],
                    grid_svc["any"],
                    grid_svc["degree"],
                    False)
                print("".join(("Melhor C = ", str(self.methods["SVM"].getC()))))
                print("".join(("Melhor Coef0 = ", str(self.methods["SVM"].getCoef0()))))
                print("".join(("Melhor Degree = ", str(self.methods["SVM"].getDegree()))))
                print("".join(("Melhor Gamma = ", str(self.methods["SVM"].getGamma()))))
                print("".join(("Melhor Kernel = ", str(self.methods["SVM"].getKernelType()))))
                print("".join(("Melhor Nu = ", str(self.methods["SVM"].getNu()))))
                print("".join(("Melhor P = ", str(self.methods["SVM"].getP()))))
            else:
                self.method_optimization(grid_svc, "SVM", False)
        elif method == "KNN":
            grid_knn = {"k": np.linspace(first_k, last_k, num=quantity_k, dtype=int)}
            self.method_optimization(grid_knn, "KNN", False)
        elif method == "MLP":
            layers = []
            for _ in range(quantity_layers):

                layer_inside = [self.mlp_layers[0],
                                [int(random.random() * range_layer) for _ in range(quantity_insidelayers)],
                                self.mlp_layers[-1]]
                if layer_inside[1] == 0:
                    layer_inside[1] = 2
                layer_inside = np.hstack(layer_inside).tolist()
                layers.append(layer_inside)
            grid_mlp = {"activation": activation,
                        "layers": layers,
                        "alpha": np.linspace(first_alpha, 100, num=quantity_alpha, dtype=float),
                        "beta": np.linspace(first_beta, 100, num=quantity_beta, dtype=float)}

            self.method_optimization(grid_mlp, "MLP", False)
        self.__init__(self.parameters["method_library"],           # Reset machine learn
                      self.parameters["library_img"],
                      "_".join((self.parameters["feature"], str(self.layer_first))),
                      self.parameters["data_base_path"],
                      constants.methods_parameters(knn_k=3, mlp_layers=[10],
                                                   svm_c=1, svm_kernel=constants.svm_kernel(inter=True),
                                                   svm_gamma=1, svm_degree=1),
                      constants.methods_selected(SVM=True, KNN=True, MLP=True)
                      )

    def labeling_only(self):
        """Da labeling of the database with train classifier """
        self.show()
        self.setup_feature()
        for index in range(len(self.images_features)):
            self.results.append(self.labeling(self.X[index], self.y[index],
                                self.y, self.images_features[index][0]))
        self.setup_save()
        self.setup_metrics()

    def run(self):
        """Do the train and classification of the database using cross validation leve-one-out"""
        self.show()
        self.setup_feature()
        self.results.append(self.cross_validation(self.X, self.y, len(self.images_features), False))
        self.setup_save()
        self.setup_metrics()


def mls_optimate(mls):
    """Run code to generate optimization graphic"""
    for ml in mls:
        methods_todo = ml.methods.keys()
        if "MLP" in methods_todo:
            ml.optimization(method="MLP", activation=["sigmoid_sym", "gaussian", "relu", "leakyrelu"],
                            quantity_layers=5, quantity_insidelayers=2, range_layer=256, quantity_alpha=2,
                            first_alpha=2.5, quantity_beta=2, first_beta=1e-2)
        elif "KNN" in methods_todo:
            ml.optimization(method="KNN", quantity_k=10, first_k=1, last_k=10)
        elif "SVM" in methods_todo:
            ml.optimization(method="SVM", svm_kernels=["linear", "poly", "rbf", "sigmoid", "chi2", "inter"],
                            quantity_C=2, first_C=0.1, quantity_gamma=1, first_gamma=0.1, quantity_degree=1,
                            first_degree=1)


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
                            mls += [MachineLearn(method_library, img_library, feature,
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
    method_libraries = constants.methods_libraries(OpenCV=True)
    img_libraries = constants.img_libraries(OpenCV=True)
    features = constants.features(histogramFull_256=[False,
                                                     256,
                                                     constants.img_processing(filterGaussianBlur=[1, ""],
                                                                              filterKMeans=[2, 10*6], HSV=[3, ""], getChannel=[4, 0])],  # [Run?, features len, img_processing**] # [order, option]
                                  histogramFilter_256=[False,
                                                       256,
                                                       constants.img_processing(HSV=[1, ""], getChannel=[2, ""],
                                                                                filterGaussianBlur=[3, ""])],
                                  histogramReduce_XXX=[True,
                                                       6*10,
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
                                  siftHistogram_XXX=[False, 6*10,
                                                     constants.img_processing(
                                                         gray=[1, ""])])
    data_base_paths = constants.data_base_paths(Data_Base_Cedulas=True, temp=False)
    methods_parameters = constants.methods_parameters(
        knn_k=3, mlp_layers=[10],
        svm_c=1, svm_kernel=constants.svm_kernel(inter=True),
        svm_gamma=1, svm_degree=1, activation="sigmoid_sym", alpha=100, beta=100)
    methods_selected = constants.methods_selected(SVM=True, KNN=True, MLP=True)
    mls_construct(todos, method_libraries, img_libraries, features,
                  data_base_paths, methods_parameters, methods_selected)
    print("".join(("Tempo de execução:", str(others.TimeConversion(time.time()-start_time)))))
