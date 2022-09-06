# Author: Fábio Campos Ferreira
# Contains step by step instructions for performing image processing, feature extraction, feature training and classification of unknown images
# Several configuration options are presented at each step for later comparison

from datetime import date
import feature_extraction
import image_processing
import classification
import result_save
import training
import time
import os


class MachineLearn:
    def __init__(self, library_index, feature_index, todo_index, data_base_path, csv_name, csv_features):
        self.library = ["OpenCV", "scikit-learn"][library_index]
        self.feature = ["histogram"][feature_index]
        self.todo = ["train", "labeling"][todo_index]
        self.data_base_path = data_base_path
        self.data_base = os.listdir(data_base_path)
        self.csv_name = csv_name
        self.csv_features = csv_features
        self.images_processed = []  # [[class,image pillow],[class,image pillow],...]
        self.images_features = []  # [[class,feature],[class,feature],...]
        self.results = []
        self.methods = {"SVM": training.SVM_create(self.library)}
        self.ml_confirm()

    def progress_bar(self, len, position_actual):
        """Print progress bar to acompaning processing"""
        print("\r")

    def ml_confirm(self):
        print("\nBiblioteca: " + self.library)
        print("Caracteristica: " + self.feature)
        print("Local das imagens: " + self.data_base_path)
        print("Fazer: " + self.todo)
        print("CSV com os resultados: " + self.csv_name)
        # TODO: show method parameters
        answer = input("Iniciar rotina de aprendizado de máquina""? (y,n) ")
        if answer == "y":
            return 0
        elif answer == "n":
            exit(1)
        else:
            print("Não entendi.")
            self.ml_confirm()

    def step_one(self):
        """Do image processing"""
        for arq in self.data_base:
            self.images_processed.append([int(os.path.basename(arq).split(".")[0]),
                                          image_processing.open_image(self.data_base_path + arq)])

    def step_two(self):
        """Do feature extractoin"""
        for img in self.images_processed:
            self.images_features.append([img[0], feature_extraction.get_features(img[1], self.feature)])
            result_save.features_save(self.csv_features, self.images_features)

    def step_three(self):
        """Do training"""
        for method in self.methods:
            y = [row[0] for row in self.images_features]
            X = [row[1] for row in self.images_features]
            training.train(X, y, method, self.methods[method], self.library)
            print("Treinamento Concluido!")

    def step_four(self):
        """Do labeling"""
        for method in self.methods:
            y = [row[0] for row in self.images_features]
            X = [row[1] for row in self.images_features]
            self.results.append(classification.labeling(X, method, self.library))

    def step_five(self):
        """Save the results"""
        result_save.save(self.csv_name, self.methods, self.results, self.library, self.feature, self.data_base_path)


if __name__ == "__main__":
    os.system("pip install -r requeriments.txt")
    # Variables of Configurations
    library_index = 0
    feature_index = 0
    todo_index = 1
    data_base_path = "../../Data_Base/Data_Base_Claro_Escuro/"
    csv_name = str(date.today())+" - Resultados"
    csv_features = str(date.today())+" - Caracteristicas"
    # Run machine learn process
    ml = MachineLearn(library_index, feature_index, todo_index, data_base_path, csv_name, csv_features)
    print("Realizando o processamento das imagens")
    ml.step_one()
    print("Extraindo as caracteristicas")
    ml.step_two()
    if ml.todo == "train":
        print("Treinando os classificadores")
        ml.step_three()
        print("Treinameto Concluido!")
    elif ml.todo == "labeling":
        print("Classificando imagens não rotuladas")
        ml.step_four()
        print("Salvando Resultdados em "+ml.csv_name)
        ml.step_five()
        print(time.perf_counter(), 'segundos')
        print("Classificação Concluida!")
