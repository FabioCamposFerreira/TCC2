# Author: Fábio Campos Ferreira
# Save the classification results generated by **classification.py** in the file **results.csv**.
from sqlite3 import Row
import numpy as np


def mls_saves(csv_name, methods, accuracy):
    """Save accuracy of the all methods in csv of the all mls builds
    """
    with open("MLS Results.csv", "+a") as csv_file:
        for method in methods:
            print("\033[91m {}\033[00m".format(""+accuracy))
            row = csv_name+";"+method+";"+accuracy
            csv_file.write("\n"+row)


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
            row_1 += ";" + method + ";"+method
            row_2 += ";" + "Rotulo Gerado;Acertou?"
        row_1 += ";" + method
        row_2 += ";" + "Tempo de classificação"
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
    arq = result[0]
    class_correct = result[1]
    class_predict = result[2]
    time = result[3]
    is_correct = str(int(str(class_correct) == str(class_predict)))
    return arq + ";" + str(class_correct)\
               + ";" + str(class_predict)\
               + ";" + is_correct\
               + ";" + str(time) + " segundos"


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
