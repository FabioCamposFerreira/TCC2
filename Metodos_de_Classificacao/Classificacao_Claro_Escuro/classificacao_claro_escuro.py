# Author: Fábio Campos
# Um exemplo simples usado para validar os classificadores dentro do aplicativp
# Recebe uma imagem pega o histograma e diz se esta claro ou escuro

import os
import time
import pickle
import treinamento_histograma_claro_escuro as treining_codes


def csv_firts_rows(csv_name, methods):
    """Create the firts lines in csv file
    Args:
        row_last : string
            Path of the image
        colum_index : string
            ??
        rows_len : int
            ??

    Returns:
        None
    """
    csv_file = open(csv_name, "w")
    row_1 = "Imagem;Classe Correta"
    row_2 = "-;-"
    for method in methods:
        row_1 += ";" + method+";"+method
        row_2 += ";" + "Rotulo Gerado;Acertou?"
        for class_ in classes:
            row_1 += ";"+method
            row_2 += ";"+"Prob. da classe " + str(class_)
    row_1 += ";"+method
    row_2 += ";"+"Tempo de classificação"
    csv_file.write(row_1+"\n")
    csv_file.write(row_2+"\n")
    csv_file.close()


def csv_write_row(csv_name,row):
    """Add row in .csv
    Args:
        csv_name : string
            Path of the csv file

    Returns:
        None
    """
    csv_file = open(csv_name, "r")
    csv_file.write(csv_file.read()+"\n"+row)
    csv_file.close()


def csv_write_last_row(csv_name):
    """Create last line (sum line) in csv

    Args:
        csv_name : string
            Path of the csv file

    Returns:
        None
    """
    csv_file = open(csv_name, "r")

    colums = csv_file.readline().split(";")
    rows_len = len(os.listdir(data_base_path))
    row_last = ""
    for colum_index in range(len(colum)):
        row_last, colum_index = make_sum_csv(row_last, colum_index, rows_len)
    csv_file.write(row_last+"\n")
    csv_file.close()
    print(time.perf_counter(), 'segundos')
    print("Classificação Concluida!")

def make_sum_csv(row_last, colum_index, rows_len):
    """Create last line (sum line) in csv

    Args:
        row_last : string
            Path of the image
        colum_index : string
            ??
        rows_len : int
            ??

    Returns:
        row_last: string
            ??
        colum_index : int
            ??
    """
    if colum_index >= 25:
        colum_index -= 26
        row_last, colum_index = make_sum_csv(
            row_last,
            colum_index,
            rows_len
        )
    if colum[colum_index] == "Acertou?" or colum[colum_index] == "Tempo de classificação":
        start_sum = chr(colum_index+65) + str(3)
        end_sum = chr(colum_index+65) + str(rows_len+2)
        row_last += ";=SUM("+start_sum+":"+end_sum+")"+"/"+str(rows_len)
    elif colum_index == 0:
        row_last += "% de acertos"
    else:
        row_last += ";-"
    return row_last, colum_index


def read_object(arq):
    """Le objeto no arquivo

    Args:
        arq: string
            localização do arquivo

    Returns:
        dump : object
            O objeto contido no arquivo
    """
    with open(arq, "rb") as f:
        dump = pickle.load(f)
        return dump


def to_rank(classifier, pattern, correct_class):
    """??

    Args:
        classifier : string
            Nome do classficador
        pattern : list 
            ??
        correct_class : int

    Returns:
        : list
            ??
    """
    st = time.time()
    clsf = read_object(classifier+".file")
    predct_proba = clsf.predict_proba([pattern])
    predct_class = predct_proba.argmax()
    return ";"+str(predct_class)+";"+str(int(str(correct_class) == str(predct_class)))+";"+str(predct_proba).replace(" ", ";").replace("[", "").replace("]", "")+";"+str(time.time()-st)+" segundos"


# Configurations
methods = treining_codes.methods
data_base_path = treining_codes.data_base_path
classes = treining_codes.classes
csv_name = "acertos_da_classficacao.csv"


csv_firts_rows(csv_name, methods)
for arq in os.listdir(data_base_path):
    row = ""
    X, y_corret = treining_codes.create_X_y(arq)
    row = arq+";"+str(y)
    print("| Classificando")
    for method in methods:
        print("| | Rotulando usando "+method)
        row += to_rank(method, X, y_corret)
    print("| Salvando Resultdados")
    csv_write_row(csv_name,row)
csv_write_last_row(csv_name)
