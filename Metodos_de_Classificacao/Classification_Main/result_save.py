# Author: Fábio Campos Ferreira
# Save the classification results generated by **classification.py** in the file **results.csv**.
from cProfile import label
from doctest import OutputChecker
from sqlite3 import Row
import numpy as np
import csv


def mls_saves(ml, csv_name: str):
    """Save accuracy of the all methods in csv of the all mls builds
    """
    with open(csv_name, "+a") as csv_file:
        for method in list(ml.methods.keys()):
            row = ('"'+ml.files_name+'"'+";"+method+";"+"accuracy_score="+str(ml.accuracy[method])+";"
                   + '"confusion_matrix='+str(ml.confusion_matrix[method])
                   + '";precision_score ='+str(ml.precision[method])+";"+"recall_score="+str(ml.recall[method]))
            csv_file.write("\n"+row)


def features_open(file_name):
    """Get features in csv file"""
    images_features = []
    with open(file_name, "r") as csv_file:
        csv_file = csv.reader(csv_file, delimiter=",",)
        for row in csv_file:
            try:
                images_features.append([int(row[0]), np.array(row[1:], dtype=np.intc).tolist()])
            except ValueError:
                pass
    return images_features


def features_save(csv_name, images_features):
    """Save features in csv file"""
    with open(csv_name, "w") as csv_file:
        row = "Classe"
        csv_file.write(row)
        for y, features in images_features:
            row = "\n"+str(y)+","+str(features).replace("[", "",).replace("]", "",)
            csv_file.write(row)


def first_row(csv_name, methods):
    """Create the firsts lines in csv file
    """
    with open(csv_name, "w") as csv_file:
        row_1 = "Imagem;Classe Correta"
        row_2 = "-;-"
        for method in methods:
            row_1 += (";" + method)*3
            row_2 += ";Rotulo Gerado;Acertou?;Tempo"
        csv_file.write(row_1)
        csv_file.write("\n"+row_2)


def read_cell(csv_name, row_pos, column_pos):
    """Return cell content in csv file
    """
    with open(csv_name, "r") as csv_file:
        for r in range(row_pos):
            columns = csv_file.readline()
        columns = csv_file.readline().split(";")
    return columns[column_pos].strip()


def make_sum(csv_name, row_last, column_index, rows_len, columns):
    """Create last line (sum line) in csv
    """
    if columns[column_index] == "Acertou?":
        sum = 0
        for row_pos in range(rows_len):
            sum += float(read_cell(csv_name, row_pos+2, column_index).replace(" segundos", ""))
        per = int(sum/rows_len*100)
        row_last += ";"+str(per)+" %"
    elif columns[column_index] == "Tempo de classificação":
        sum = 0
        for row_pos in range(rows_len):
            sum += float(read_cell(csv_name, row_pos+2, column_index).replace(" segundos", ""))
        per = sum/rows_len
        row_last += ";"+str(per)+" segundos"
    elif column_index == 0:
        row_last += "% de acertos"
    else:
        row_last += ";-"
    return row_last, column_index


def predict_line(result):
    """Construct one line for one result of the classifier
    """
    line = result.pop(0)+";"+result.pop(0)  # arq name + # class_correct
    for r in range(len(result)):
        if r % 3 == 2:
            line += ";"+result[r]  # class_predict
            line += ";"+str(int(str(result[r-1]) == str(result[r])))  # is_correct
        elif r % 3 == 0 and r != 0:
            line += ";"+str(result[r])+" segundos"  # time
    return line


def write_row(csv_name, row):
    """Add new row in .csv

    """
    with open(csv_name, "a") as csv_file:
        csv_file.write("\n"+row)


def construct_last_row(csv_name):
    """Create last line (sum line) in csv
    """
    with open(csv_name, "r") as csv_file:
        csv_file.readline()
        columns = csv_file.readline().strip().split(";")
        rows_len = len(csv_file.read().split("\n"))
    row_last = ""
    for column_index in range(len(columns)):
        row_last, column_index = make_sum(csv_name,
                                          row_last,
                                          column_index,
                                          rows_len,
                                          columns)
    return row_last


def save(csv_name: str, methods: list, results: list):
    """Construct csv file with results for each method od the classification"""
    first_row(csv_name, methods)
    for result in results:
        row = predict_line(result)
        write_row(csv_name, row)
    row = construct_last_row(csv_name)
    write_row(csv_name, row)