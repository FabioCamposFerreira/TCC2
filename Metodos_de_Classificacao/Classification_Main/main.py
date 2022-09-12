# Author: Fábio Campos Ferreira
# Contains step by step instructions for performing image processing, feature extraction, feature training and classification of unknown images
# Several configuration options are presented at each step for later comparison

from operator import index
import install_dev
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    def __init__(self, library: str, library_img: str, feature: str, data_base_path: str):
        self.library = library
        self.library_img = library_img
        self.feature = feature
        self.data_base_path = data_base_path
        self.data_base = os.listdir(data_base_path)
        self.csv_name = str(
            date.today())+" Resultados com "+self.library+" - "+self.library_img+" - "+self.feature+".csv"
        self.csv_features = self.csv_name.replace(" Resultados com ", " Característica com ")
        self.images_processed = []  # [[class,image pillow],[class,image pillow],...]
        self.images_features = []  # [[class,feature],[class,feature],...]
        self.results = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.imgs_test = []
        self.accuracy = []
        self.methods = {"SVM": training.SVM_create(self.library)}

    def run(self):
        self.ml_show()
        """Train or classify"""
        print("Realizando o processamento das imagens")
        self.step_one()
        print("Extraindo as características")
        self.step_two()
        print("Treinando os classificadores")
        self.step_three()
        print("Classificando imagens não rotuladas")
        self.step_four()
        print("Salvando Resultados em "+self.csv_name)
        self.step_five()
        print(time.perf_counter(), 'segundos')
        print("Classificação Concluída!")

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

    def ml_show(self):
        print("\nBiblioteca de classificação: " + "\033[91m {}\033[00m".format(self.library))
        print("Biblioteca de processamento de imagem: " + "\033[91m {}\033[00m".format(self.library_img))
        print("Característica: " + "\033[91m {}\033[00m".format(self.feature))
        print("Local das imagens: " + "\033[91m {}\033[00m".format(self.data_base_path))
        print("CSV com os resultados: " + "\033[91m {}\033[00m".format(self.csv_name))

    def step_one(self):
        """Do image processing"""
        total = len(self.data_base)
        actual = 0
        for arq in self.data_base:
            actual += 1
            self.progress_bar(actual, total)
            self.images_processed.append([os.path.basename(arq), image_processing.open_image(
                arq=self.data_base_path + arq, inverted=False, library_img=self.library_img)])
            self.images_processed.append([os.path.basename(arq)+" (Inverted)",
                                          image_processing.open_image(arq=self.data_base_path+arq,
                                                                      inverted=True,
                                                                      library_img=self.library_img)])
        self.images_processed.sort()

    def step_two(self):
        """Do feature extraction"""
        total = len(self.images_processed)
        actual = 0
        for img in self.images_processed:
            actual += 1
            self.progress_bar(actual, total)
            self.images_features.append([int(img[0].split(".")[0]),
                                         feature_extraction.get_features(img[1], self.feature, self.library_img)])
        result_save.features_save(self.csv_features, self.images_features)

    def step_three(self):
        """Do training"""
        total = len(self.methods)
        actual = 0

        y = [row[0] for row in self.images_features]
        X = [row[1] for row in self.images_features]
        print("\033[92m {}\033[00m".format(type(X[0][0])))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        for method in self.methods:
            actual += 1
            self.progress_bar(actual, total)
            training.train(self.X_train, self.y_train, method, self.methods[method], self.library)
            print("Treinamento Concluído!")

    def step_four(self):
        """Do labeling"""
        total = len(self.methods)
        actual = 0
        img_names = [row[0] for row in self.images_processed]
        _, self.imgs_test = train_test_split(img_names, test_size=0.2, random_state=0)
        for method in self.methods:
            actual += 1
            self.progress_bar(actual, total)
            start_time = time.time()
            class_predictions = classification.labeling(self.X_test, method, self.library)
            end_time = (time.time()-start_time)/len(class_predictions)
            self.accuracy += [accuracy_score(self.y_test, class_predictions)]
            for index in range(len(self.imgs_test)):
                self.results.append([self.imgs_test[index], self.y_test[index],
                                    int(class_predictions[index]), end_time])

    def step_five(self):
        """Save the results"""
        self.results = np.array(self.results)
        result_save.save(self.csv_name, self.methods, self.results)


if __name__ == "__main__":
    mls = []
    # Create machine learn 1 (ml)
    # Library: OpenCV,scikit-learn
    # Library_img: OpenCV, Pillow
    # Features: histogram
    # data_base_path: images have name with class "className.*". Ex.: 1.1.png
    # mls += [MachineLearn(library="OpenCV", library_img="Pillow", feature="histogram",
                        #  data_base_path="../../Data_Base/temp/")]
    mls += [MachineLearn(library="OpenCV", library_img="OpenCV", feature="histogram",
                         data_base_path="../../Data_Base/temp/")]
    # Run machine learns
    for ml in mls:
        ml.run()
        result_save.mls_saves(ml.csv_name, list(ml.methods.keys()), ml.accuracy)
