# Author: Fábio Campos Ferreira
# Contains step by step instructions for performing image processing, feature extraction, feature training and classification of unknown images
# Several configuration options are presented at each step for later comparison

from textwrap import indent
import install_dev
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import os
import feature_extraction
import image_processing
import classification
import result_save
import training


class MachineLearn:
    def __init__(self, library: str, library_img: str, feature: str, data_base_path: str, knn_k=3, mlp_layers=[10]):
        self.data_base = os.listdir(data_base_path)
        self.data_base.sort()
        self.parameters = {
            # Global
            "data_base_path": data_base_path,
            "library": library,
            "library_img": library_img,
            "feature": feature,
            # To SVM
            # To KNN
            "knn_k": knn_k,
            # To MLP
            "mlp_layers": (  # [fist layer]+[middle layers]+[last layers]
                [256 if feature == "histogram" else None]
                + mlp_layers
                + [len(list(dict.fromkeys([arq.split(".")[0] for arq in self.data_base])))])
        }
        # Results
        self.images_processed = []  # [[name,image],[name,image],...]
        self.images_features = []  # [[name,feature],[name,feature],...]
        self.results = []
        self.accuracy = {}
        self.precision = {}
        self.confusion_matrix = {}
        self.recall = {}
        # Files with results
        self.path_output = "./Output/"
        self.path_classifiers = self.path_output+"Classifiers/"
        self.path_features = self.path_output+"Patterns/"
        self.path_graphics = (
            self.path_output
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
        self.methods = {
            "SVM": training.SVM_create(self.parameters["library"]),
            "KNN": training.KNN_create(self.parameters["library"], self.parameters["knn_k"]),
            "MLP": training.MLP_create(self.parameters["library"], self.parameters["mlp_layers"])}

    def progress_bar(self, actual, total):
        """Print progress bar to accompanying processing"""
        line_width = int(subprocess.check_output("tput cols", shell=True))
        line_structure = "[] 100%"
        bar_len = (line_width-len(line_structure))
        hash_quantity = int(actual/total*bar_len)
        hyphen_quantity = bar_len-hash_quantity
        line = "[{}] {}%".format("#"*hash_quantity+"-"*hyphen_quantity, int(actual/total*100))
        print(line, end="\r")

    def show(self):
        """Show the classifications parameters"""
        print("\nParametros usados: ")
        for parameter in self.files_name.replace("./results/XXX-", "").split(","):
            print("\t\033[91m {}\033[00m".format(parameter))
        print("CSV com as características: " + "\033[91m {}\033[00m".format(self.csv_features))
        print("CSV com os resultados: " + "\033[91m {}\033[00m".format(self.csv_results))

    def setup_images(self):
        """Do image processing"""
        print("Realizando o processamento das imagens")
        total = len(self.data_base)
        actual = 0
        for arq in self.data_base:
            actual += 1
            self.progress_bar(actual, total)
            self.images_processed.append(
                [arq, image_processing.open_image(
                    self.parameters["data_base_path"]
                    + arq, self.parameters["library_img"])])
            self.images_processed.append(
                [arq+" (Inverted)",
                 image_processing.open_image(
                     self.parameters["data_base_path"]+arq,
                     self.parameters["library_img"],
                     inverted=True)])
        print(end='\x1b[2K')  # clear progress bar

    def setup_feature(self):
        """Do feature extraction"""
        try:
            self.images_features = result_save.features_open(self.csv_features)
        except:
            self.images_features = []
            self.setup_images()
            print("Extraindo as características")
            total = len(self.images_processed)
            actual = 0
            for img in self.images_processed:
                actual += 1
                self.progress_bar(actual, total)
                self.images_features.append(
                    [img[0], feature_extraction.get_features(
                        img[1],
                        self.parameters["feature"],
                        self.parameters["library_img"])])
            print(end='\x1b[2K')  # clear progress bar
            result_save.features_save(self.csv_features, self.images_features)
            print("Salvando gráficos em "+self.path_graphics)
        result_save.graphics_save(self.path_graphics, self.images_features)

    def setup_train(self, X, y):
        """Do training"""
        for method in self.methods:
            training.train(
                X, y, method, self.methods[method],
                self.parameters["library"],
                self.xml_name)

    def labeling(self, X: str, y_correct: int, y_full: list, img_name: str):
        """Do labeling and update results"""
        result = [img_name, y_correct]
        for method in self.methods:
            start_time = time.time()
            y_predict = classification.labeling(X, y_full, method, self.parameters["library"], self.xml_name)
            result += [int(y_predict), time.time()-start_time]
        self.results.append(result)

    def setup_metrics(self):
        """Generate metrics of the result classification"""
        results = np.array(self.results)
        classes_correct = results[:, 1]
        classes_correct = np.array(classes_correct, dtype=int)
        results = results[:, 2:]  # remove image name and class correct
        results = np.array(results, dtype=float)
        results = np.array(results, dtype=int)
        for index, method in enumerate(self.methods):
            self.accuracy[method] = accuracy_score(classes_correct, results[:, (2*index)])
            self.precision[method] = precision_score(classes_correct, results[:, (2*index)],
                                                     average="weighted", sample_weight=classes_correct)
            self.confusion_matrix[method] = confusion_matrix(classes_correct, results[:, (2*index)])
            self.recall[method] = recall_score(classes_correct, results[:, (2*index)], average="weighted",
                                               sample_weight=classes_correct, zero_division=0)

    def setup_save(self):
        """Save the results fo the labeling"""
        print("Salvando Resultados em "+self.csv_results)
        result_save.save(self.csv_results, self.methods,  np.array(self.results))

    def run(self):
        """Do the train and classification of the database using cross validation leve-one-out"""
        self.show()
        self.setup_feature()
        y = [int(row[0].split(".")[0]) for row in self.images_features]
        X = [row[1] for row in self.images_features]
        features_len = len(self.images_features)
        print("Realizando o treinamento e classificação usando cross validation leve-one-out")
        for index in range(features_len):
            self.progress_bar(index, features_len)
            self.setup_train(X[:index]+X[index+1:], y[:index]+y[index+1:])
            self.labeling(X[index], y[index], y, self.images_features[index][0])
        print(end='\x1b[2K')  # clear progress bar
        self.setup_save()
        self.setup_metrics()


if __name__ == "__main__":
    """
    Library: OpenCV,scikit-learn
    Library_img: Pillow
    Features: histogram
    data_base_path: images have name with class "className.*". Ex.: 1.1.png
    knn_k: any int >=1
    """
    mls = []
    mls += [MachineLearn(library="OpenCV", library_img="Pillow", feature="histogram",
                         data_base_path="../../Data_Base/Data_Base_Cedulas/")]
    mls += [MachineLearn(library="scikit-learn", library_img="Pillow", feature="histogram",
                         data_base_path="../../Data_Base/Data_Base_Cedulas/")]
    # Run machine learns
    for ml in mls:
        ml.run()
        result_save.mls_saves(ml, ml.path_output+"MLS Results.csv")
    print(time.perf_counter(), 'segundos')
# TODO: criar função para escolher os melhores parâmetros de cada classificador
