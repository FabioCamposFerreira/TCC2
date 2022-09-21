# Author: Fábio Campos Ferreira
# Save the classification results generated by **classification.py** in the file **results.csv**.
from bokeh.palettes import Dark2_5 as palette
from scipy import stats
import matplotlib.pyplot as plt
import bokeh.plotting as bokeh
import pandas as pd
import numpy as np
import itertools
import csv
import os


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
                images_features.append([row[0], np.array(row[1:], dtype=np.intc).tolist()])
            except ValueError:
                pass
    return images_features


def graphics_interactive(curves: list, labels: list, file_path: str):
    """Save in file html graph interactive with many curves"""
    colors = itertools.cycle(palette)
    f = bokeh.figure(sizing_mode="stretch_both", output_backend="svg")
    for color, curve, label in zip(colors, curves, labels):
        f.line(range(len(curve)), curve,  color=color, legend_label=label, line_width=2)
    f.legend.location = "top_right"
    f.legend.click_policy = "hide"
    # f.output_backend = "svg"
    # f.sizing_mode = "stretch_both"
    bokeh.output_file(file_path+".html")
    bokeh.save(f)


def graphics_splom(classes, labels, features, files_name):
    """"""
    df = pd.DataFrame(data=d)

    pass


def graphics_box2(classes: set, labels: list[str], features: np.ndarray, file_path: str):
    """Construct and save box chart to every class separate by feature"""
    pass
    _, axs = plt.subplots(len(features[0], figsize=(len(classes)/3, 10)))
    for c in enumerate(classes):
        positions = list(labels[:] == c)
        axs[index].boxplot(features[positions])
        axs[index].set(ylabel="Notas de "+c)
    plt.savefig(file_path.replace("XXX", "Box2")+".pdf", bbox_inches='tight')


def graphics_box1(classes: set, labels: list[str], features: np.ndarray, file_path: str):
    """Construct and save box chart to every feature separate by class"""
    _, axs = plt.subplots(len(classes), figsize=(len(features[0])/3, 10))
    for index, c in enumerate(classes):
        positions = list(labels[:] == c)
        axs[index].boxplot(features[positions])
        axs[index].set(ylabel="Notas de "+c)
    plt.savefig(file_path.replace("XXX", "Box1")+".pdf", bbox_inches='tight')


def graphics_lines(classes: set, labels: list[str],  features: list[list[int]], file_path: str):
    legends = {"mean": [], "median": [], "mode": [], "Sandard Deviation": []}
    curves = {"mean": [], "median": [], "mode": [], "Sandard Deviation": []}
    for c in classes:
        positions = list(labels[:] == c)
        for key in legends.keys():
            if key == "mean":
                curves[key].append(np.mean(features[positions], axis=0))
            elif key == "median":
                curves[key].append(np.median(features[positions], axis=0))
            elif key == "mode":
                curves[key].append(stats.mode(features[positions], axis=0)[0][0])
            elif key == "Sandard Deviation":
                curves[key].append(np.std(features[positions], axis=0))
            legends[key].append("Nota de "+str(c))
    for key in legends.keys():
        graphics_interactive(curves[key], legends[key], file_path.replace("XXX", key))


def graphics_save(files_name: str, images_features: list):
    """Save graphics of the features to compare quality of distribution"""
    features = []
    labels = np.array([])
    for item in images_features:
        labels = np.append(labels, item[0].split(".")[0])
        features.append(item[1])
    features = np.array(features)
    classes = set(labels)
    graphics_lines(classes, labels, features, files_name)
    graphics_box1(classes, labels, features, files_name)
    graphics_box2(classes, labels, features, files_name)
    graphics_splom(classes, labels, features, files_name)


def features_save(csv_name: str, images_features: list):
    """Save features in csv file"""
    os.makedirs(os.path.dirname(csv_name), exist_ok=True)
    with open(csv_name, "w") as csv_file:
        row = "Nome,Classe"
        csv_file.write(row)
        for image, feature in images_features:
            row = "\n"+str(image)+","+str(feature).replace("[", "",).replace("]", "",)
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
    class_correct = result[1]
    line = result[0]+";"+class_correct  # arq name + class_correct
    result = result[2:]
    for r in range(len(result)):
        if r % 2 == 0:
            line += ";"+result[r]  # class_predict
            line += ";"+str(int(class_correct == str(result[r])))  # is_correct
        else:
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
