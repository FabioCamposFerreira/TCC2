# Author: Fábio Campos
# Um exemplo simples usado para validar os classificadores dentro do aplicativp
# Recebe uma imagem pega o histograma e diz se esta claro ou escuro

import os
import time
import pickle
import treinamento_histograma_claro_escuro as treining_codes
import cv2 as cv
import numpy as np
import csv_construct


def read_object(arq):
    """Le objeto no arquivo

    Args:
        arq: string
            localização do arquivo

    Returns:
        dump : object
            O objeto contido no arquivo
    """
    return cv.ml_SVM.load(arq)


def to_rank(classifier, pattern, correct_class):
    """??

    Args:
        classifier : string
            Nome do classficador
        pattern : list 
            ??
        correct_class : int

    Returns:
        : str
            ??
    """
    st = time.time()
    clsf = read_object(classifier+".file")
    pattern = np.matrix(pattern, dtype=np.float32)
    predict_class = clsf.predict(pattern)[1][0][0]
    delta_time = time.time() - st
    return 


# Configurations
methods = treining_codes.methods
data_base_path = treining_codes.data_base_path
classes = treining_codes.classes
csv_name = "acertos_da_classficacao.csv"
results = ""

print("| Classificando")

for arq in os.listdir(data_base_path):
    row = ""
    X, y_corret = treining_codes.create_X_y(data_base_path+arq)
    row = arq+";"+str(y_corret)
    for method in methods:
        print("| | Rotulando usando "+method)
        results.append(to_rank(method, X, y_corret))
    print("| Salvando Resultdados")
    
csv_construct.firts_rows(csv_name, methods)
row = csv_construct.predict_line(predict_class, correct_class, delta_time)
csv_construct.write_row(csv_name, row)
csv_construct.write_last_row(csv_name)
print(time.perf_counter(), 'segundos')
print("Classificação Concluida!")

