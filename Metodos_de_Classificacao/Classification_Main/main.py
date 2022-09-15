# Author: Fábio Campos Ferreira
# Contains step by step instructions for performing image processing, feature extraction, feature training and classification of unknown images
# Several configuration options are presented at each step for later comparison

import install_dev
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
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
        self.images_processed = []  # [[class,image],[class,image],...]
        self.images_features = []  # [[class,feature],[class,feature],...]
        self.results = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.imgs_test = []
        self.accuracy = {}
        self.precision = {}
        self.confusion_matrix = {}
        self.recall = {}
        # Files with results
        self.files_name = (
            "./results/XXX-"
            + ",".join(p + "=" + str(self.parameters[p]).split("/")[-2:][0] for p in self.parameters))
        self.csv_results = self.files_name.replace("XXX", "Resultados")+".csv"
        self.csv_features = self.files_name.replace("XXX", "Características")+".csv"
        self.xml_name = self.files_name+".xml"
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
        print(end='\x1b[2K')

    def show(self):
        print("\nParametros usados: ")
        for parameter in self.files_name.replace("./results/XXX-", "").split("-"):
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
                    + arq, self.parameters["library_img"])
                 ])
            self.images_processed.append(
                [arq+" (Inverted)",
                 image_processing.open_image(
                     self.parameters["data_base_path"]+arq,
                     self.parameters["library_img"],
                     inverted=True)
                 ])

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
                    [
                        int(img[0].split(".")[0]),
                        feature_extraction.get_features(
                            img[1],
                            self.parameters["feature"],
                            self.parameters["library_img"]
                        )
                    ]
                )
            result_save.features_save(self.csv_features, self.images_features)

    def setup_train(self):
        """Do training"""
        print("Treinando os classificadores")
        total = len(self.methods)
        actual = 0
        y = [row[0] for row in self.images_features]
        X = [row[1] for row in self.images_features]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        for method in self.methods:
            actual += 1
            self.progress_bar(actual, total)
            training.train(
                self.X_train, self.y_train, method, self.methods[method],
                self.parameters["library"],
                self.xml_name)
        print("Treinamento Concluído!")

    def setup_label(self):
        """Do labeling"""
        if len(self.images_features) == 0:
            self.setup_feature()
        self.setup_train()
        print("Classificando imagens não rotuladas")
        total = len(self.methods)
        actual = 0
        _, self.imgs_test = train_test_split(self.data_base, test_size=0.2, random_state=0)
        for index in range(len(self.imgs_test)):
            self.results.append([self.imgs_test[index]])
        for method in self.methods:
            actual += 1
            self.progress_bar(actual, total)
            start_time = time.time()
            class_predictions = classification.labeling(self.X_test, method, self.parameters["library"], self.xml_name)
            end_time = (time.time()-start_time)/len(class_predictions)
            # invert one-hot enconding to MLP
            if method == "MLP" and self.parameters["library"]=="OpenCV":
                enc = OneHotEncoder(sparse=False, dtype=np.float32, handle_unknown="ignore")
                _ = enc.fit_transform(np.array(self.y_test).reshape(-1, 1))
                class_predictions = enc.inverse_transform(class_predictions)
                class_predictions[class_predictions == None] = 0
                class_predictions = np.squeeze(class_predictions).tolist()
            self.accuracy[method] = accuracy_score(self.y_test, class_predictions)
            self.precision[method] = precision_score(
                self.y_test, class_predictions,
                average="weighted", sample_weight=self.y_test)
            self.confusion_matrix[method] = confusion_matrix(self.y_test, class_predictions)
            self.recall[method] = recall_score(
                self.y_test, class_predictions,
                average="weighted", sample_weight=self.y_test, zero_division=0)
            for index in range(len(self.imgs_test)):
                self.results[index] += [self.y_test[index], int(class_predictions[index]), end_time]
        print("Classificação Concluída!")
        self.setup_save()

    def setup_save(self):
        """Save the results fo the labeling"""
        print("Salvando Resultados em "+self.csv_results)
        self.results = np.array(self.results)
        result_save.save(self.csv_results, self.methods, self.results)


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
        ml.setup_label()
        result_save.mls_saves(ml)
    print(time.perf_counter(), 'segundos')
# TODO: criar função para escolher os melhores parâmetros de cada classificador
