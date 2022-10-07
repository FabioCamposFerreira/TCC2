# Author: Fábio Campos Ferreira
# Contains step by step instructions for performing image processing, feature extraction, feature training and classification of unknown images
# Several configuration options are presented at each step for later comparison

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, mean_squared_error
import numpy as np
import install_dev
import time
import os
import feature_extraction
import image_processing
from multiprocessing import Process, Manager
import classification
import result_save
import training
import random
import others


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
            "SVM": training.SVM_create(library=self.parameters["library"]),
            "KNN": training.KNN_create(library=self.parameters["library"],
                                       k=self.parameters["knn_k"]),
            "MLP": training.MLP_create(
                library=self.parameters["library"],
                mlp_layers=self.parameters["mlp_layers"])}

    def show(self):
        """Show the classifications parameters"""
        print("\nParâmetros usados: ")
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
            image_processing.progress_bar(actual, total)
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
                image_processing.progress_bar(actual, total)
                self.images_features.append(
                    [img[0], feature_extraction.get_features(
                        img[1],
                        self.parameters["feature"],
                        self.parameters["library_img"])])
            print(end='\x1b[2K')  # clear progress bar
            result_save.features_save(self.csv_features, self.images_features)
            print("Salvando gráficos em "+self.path_graphics)
            result_save.graphics_save(self.path_graphics, self.images_features)
        self.y = [int(row[0].split(".")[0]) for row in self.images_features]
        self.X = [row[1] for row in self.images_features]

    def setup_train(self, X: list[float], y: list[int], file_save: True):
        """Do training and save classifiers in files"""
        classifier = {}
        for method in self.methods:
            classifier[method] = training.train(X, y, method, self.methods[method],
                                                self.parameters["library"],
                                                self.xml_name, file_save)
        return classifier

    def labeling(self, X: str, y_correct: int, y_full: list, img_name: str, classifier):
        """Do labeling and update results"""
        result = [img_name, y_correct]
        for method in self.methods:
            start_time = time.time()
            y_predict = classification.labeling(X, y_full, method, self.parameters["library"],
                                                self.xml_name, classifier[method])
            result += [int(y_predict), time.time()-start_time]
        return result

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
                                                     average="weighted", sample_weight=classes_correct, zero_division=0)
            self.confusion_matrix[method] = confusion_matrix(classes_correct, results[:, (2*index)])
            self.recall[method] = recall_score(classes_correct, results[:, (2*index)], average="weighted",
                                               sample_weight=classes_correct, zero_division=0)
            self.meansquare_error[method] = mean_squared_error(classes_correct, results[:, (2 * index)],
                                                               sample_weight=classes_correct)

    def setup_save(self):
        """Save the results fo the labeling"""
        print("Salvando Resultados em "+self.csv_results)
        result_save.save(self.csv_results, self.methods,  np.array(self.results))

    def validation(self, X: list[list[float]], y: list[int], index: int, results: dict = None):
        """Train and classify data to one validation in cross validation"""
        classifier = self.setup_train(X[:index]+X[index+1:], y[:index]+y[index+1:], file_save=False)
        if results == None:
            return self.labeling(X[index], y[index], y, self.images_features[index][0], classifier=classifier)
        else:
            results.append(self.labeling(X[index], y[index], y, self.images_features[index][0], classifier=classifier))

    def validation_parallel(self,  X: list[list[float]], y: list[int], features_len: int):
        """Run multiples self.validation in parallel"""
        progress_bar = others.ProgressBar("Fazendo validação cruzada", features_len, 0)
        processes = []
        results = Manager().list()
        for index in range(features_len):
            p = Process(target=self.validation, args=(X, y, index, results))
            processes.append(p)
            p.start()
        for index, process in enumerate(processes):  # waiting finish all process
            progress_bar.print(index)
            process.join()
        progress_bar.end()
        self.results = list(results)

    def validation_serial(self,  X: list[list[float]], y: list[int], features_len: int):
        """Run multiples self.validation in serial"""
        progress_bar = others.ProgressBar("Fazendo validação cruzada", features_len, 0)
        results = []
        for index in range(features_len):
            progress_bar.print(index)
            results.append(self.validation(X, y, index))
        progress_bar.end()
        self.results = list(results)

    def cross_validation(self, X, y, features_len, parallel: bool = True):
        """Make cross validation one-leave-out to each method"""
        if 0:
            print("Realizando o treinamento e classificação usando cross validation leve-one-out")
        if parallel == True:
            self.validation_parallel(X, y, features_len)
        else:
            self.validation_serial(X, y, features_len)

    def cross_validation_parallel(self, X, y, features_len, method, parameters, results_xylabel):
        """Call self.cross validation to run in parallel form"""
        self.cross_validation(X, y, features_len, False)
        self.setup_metrics()
        results_xylabel['x'] += [self.accuracy[method]]
        results_xylabel['y'] += [self.precision[method]]
        results_xylabel['labels'] += [str(parameters)]

    def parameters_combination(self, keys: list[str], grid: dict, parameters: dict, method: str,
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
                progress_bar.print(actual, line=-1)
            if method == "SVM":
                self.methods = {"SVM": training.SVM_create(self.parameters["library"],
                                                           C=parameters["C"],
                                                           kernel=parameters["kernel"],
                                                           gamma=parameters["gamma"],
                                                           degree=parameters["degree"])}
            elif method == "KNN":
                self.methods = {"KNN": training.KNN_create(self.parameters["library"], k=parameters["k"])}
            elif method == "MLP":
                self.methods = {"MLP": training.MLP_create(library=self.parameters["library"],
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
        results_xylabel = Manager().dict({"x": [], "y": [], "labels": []})
        print("Otimizando {}: ".format(method)+str(grid).replace("{", "").replace("}", ""))
        total = 1
        for key in grid.keys():
            total *= len(grid[key])
        print("\n")  # to progress bar not bug
        progress_bar = others.ProgressBar("Otimizando "+method, total, -1)
        if parallel == True:
            process_parallel = []
            self.parameters_combination(keys=list(grid.keys()),
                                        grid=grid, parameters={},
                                        method=method, progress_bar=None, actual=0, results_xylabel=results_xylabel,
                                        process_parallel=process_parallel)
            for index, p in enumerate(process_parallel):
                progress_bar.print(index)
                p.join()
        progress_bar.end()
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
                        "gamma": np.linspace(first_gamma, 100, num=quantity_gamma, dtype=float),
                        "C": np.linspace(first_C, 1000, num=quantity_C, dtype=float),
                        "degree": np.linspace(first_degree, 10, num=quantity_degree, dtype=int)}
            self.method_optimization(grid_svc, "SVM")
        elif method == "KNN":
            grid_knn = {"k": np.linspace(first_k, last_k, num=quantity_k, dtype=int)}
            self.method_optimization(grid_knn, "KNN")
        elif method == "MLP":
            layers = []
            for _ in range(quantity_layers):

                layer_inside = [self.parameters["mlp_layers"][0],
                                [int(random.random() * range_layer) for _ in range(quantity_insidelayers)],
                                self.parameters["mlp_layers"][-1]]
                if layer_inside[1] == 0:
                    layer_inside[1] = 2
                layer_inside = np.hstack(layer_inside).tolist()
                layers.append(layer_inside)
            grid_mlp = {"activation": activation,
                        "layers": layers,
                        "alpha": np.linspace(first_alpha, 100, num=quantity_alpha, dtype=float),
                        "beta": np.linspace(first_beta, 100, num=quantity_beta, dtype=float)}

            self.method_optimization(grid_mlp, "MLP")
        self.__init__(self.parameters["library"],           # Reset machine learn
                      self.parameters["library_img"],
                      self.parameters["feature"],
                      self.parameters["data_base_path"])

    def run(self):
        """Do the train and classification of the database using cross validation leve-one-out"""
        self.show()
        self.setup_feature()
        self.results.append(self.cross_validation(self.X, self.y, len(self.images_features), False))
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
    if 0:
        mls += [MachineLearn(library="scikit-learn", library_img="Pillow", feature="histogram",
                             data_base_path="../../Data_Base/Data_Base_Cedulas/")]
    # Run machine learns
    for ml in mls:
        if 1:  # otimization
            # ml.optimization(method="MLP", activation=["sigmoid_sym", "gaussian", "relu", "leakyrelu"],
            #                 quantity_layers=1, quantity_insidelayers=1, range_layer=10, quantity_alpha=1,
            #                 first_alpha=2.5, quantity_beta=1, first_beta=1e-2)
            ml.optimization(method="SVM", svm_kernels=["linear", "poly", "rbf", "sigmoid", "chi2", "inter"],
                            quantity_C=1, first_C=0.1, quantity_gamma=1, first_gamma=0.1, quantity_degree=1,
                            first_degree=1)
            # ml.optimization(method="KNN", quantity_k=5, first_k=1, last_k=10)
        else:
            ml.run()
            result_save.mls_saves(ml, ml.path_output+"MLS Results.csv")
    print(time.perf_counter(), 'segundos')

# TODO: criar função para escolher os melhores parâmetros de cada classificador
