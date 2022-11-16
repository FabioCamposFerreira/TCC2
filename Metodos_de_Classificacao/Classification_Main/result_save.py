# Author: Fábio Campos Ferreira
# Save the classification results generated by **classification.py** in the file **results.csv**.
import csv
import itertools
import math
import os
from typing import List

import bokeh.plotting as bokeh
import cv2 as cv
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.models import LabelSet, HoverTool, CustomJS, Dropdown
from bokeh.palettes import Dark2_5 as palette
from bokeh.layouts import column, row
from bokeh.transform import factor_cmap
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import pandas as pd

import others


def optimization_graph(points: dict, file_path: str):
    """Save graph 2d with points and one legend to each point

    Parameters
    ----------
    points : dict
        keys are accuracy and parameters
    """
    menu = []
    labels = ""
    points["x"] = points["accuracy"]
    points["y"] = points["accuracy"]
    key_strings = ""
    keys = points.keys()
    # contruct click menu
    for key in keys:
        type_test = type(points[key][0])
        if type_test != str and type_test != list:
            if key != "x" and key != "y":
                menu += [(key, key)]
        else:
            key_strings = key
        if key != "x" and key != "y":
            labels += "".join((",", key, ":@", key))
    # select legend id
    if "kernel" in keys:
        legend_id = "kernel"
    elif "activation" in keys:
        legend_id = "activation"
    else:
        legend_id = "k"
    # construct graph
    source = pd.DataFrame.from_dict(points)
    try:
        index_cmap = factor_cmap(key_strings, palette=palette, factors=sorted(source[key_strings].unique()))
    except:
        index_cmap = "red"
    source = bokeh.ColumnDataSource(points)
    TOOLTIPS = [("(x,y)", "($x, $y)"), ("label", labels)]
    f = bokeh.figure(sizing_mode="stretch_both", output_backend="svg", tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     tooltips=TOOLTIPS)
    f.circle("x", "y", source=source, legend_field=legend_id, size=10, color=index_cmap)
    f.legend.click_policy = "hide"
    f.add_layout(f.legend[0], 'right')
    f.xaxis.axis_label = "accuracy"
    f.yaxis.axis_label = "accuracy"
    dropdown_x = Dropdown(label="Select X", button_type="warning", menu=menu)
    dropdown_y = Dropdown(label="Select Y", button_type="warning", menu=menu)
    dropdown_x.js_on_event("menu_item_click", CustomJS(args=dict(source=source, x_label=f.xaxis[0]), code="""
    const new_data = Object.assign({}, source.data)
    new_data.x = source.data[this.item]
    source.data = new_data
    console.log(x_label.axis_label)
    x_label.axis_label = this.item
    """))
    dropdown_y.js_on_event("menu_item_click", CustomJS(args=dict(source=source, y_label=f.yaxis[0]), code="""
    const new_data = Object.assign({}, source.data)
    new_data.y = source.data[this.item]
    source.data = new_data
    y_label.axis_label = this.item
    """))
    bokeh.output_file(file_path+'.html')
    bokeh.save(column(row(dropdown_x, dropdown_y, sizing_mode="scale_width"), f, sizing_mode="stretch_both"))


def mls_saves(ml, csv_name: str):
    """Save accuracy of the all methods in csv of the all mls builds
    """
    with open(csv_name, "+a") as csv_file:
        for method in list(ml.methods.keys()):
            row = ('"' + ml.files_name.replace("XXX", "Resultados") + '"' + ";" + method + ";" + "accuracy_score="
                   + str(ml.accuracy[method]) + ";" + '"confusion_matrix=' + str(ml.confusion_matrix[method])
                   + '";precision_score =' + str(ml.precision[method]) + ";" + "recall_score=" + str(ml.recall[method])
                   + ";"
                   + "meansquare_error=" + str(ml.meansquare_error[method]))
            csv_file.write("\n"+row)


def features_open(file_name):
    """Get features in csv file"""
    images_features = []
    with open(file_name, "r") as csv_file:
        csv_file = csv.reader(csv_file, delimiter=",",)
        for row in csv_file:
            try:
                images_features.append([row[0], np.array(row[1:], dtype=float)])
            except ValueError:
                pass
    return images_features


def add_hue_bar(f: bokeh.Figure, length: int):
    """Add in graphic from bokeh bar with hue spectrum"""
    x = [r for r in range(length)]
    y = [-1 for _ in range(length)]
    hsv = [np.uint8([[[hue, 255, int(255/2)]]]) for hue in range(0, 256, int(256/length))]
    rgb = [cv.cvtColor(hsv_one, cv.COLOR_HSV2RGB_FULL)[0][0] for hsv_one in hsv]
    hex_ = ['#%02x%02x%02x' % tuple(rgb_one.tolist()) for rgb_one in rgb]
    f.square(x[0:length], y[0:length], size=20, color=hex_[0:length])
    return f


def graphics_interactive(curves: list, labels: list, file_path: str):
    """Save in file html graph interactive with many curves"""
    colors = itertools.cycle(palette)
    length = len(curves[0])
    x = range(length)
    f = bokeh.figure(sizing_mode="stretch_both", tools="pan,wheel_zoom,box_zoom,reset,save", output_backend="svg")
    for curve, label, color in zip(curves, labels, colors):
        l = f.line(x, curve,  line_color=color, legend_label=label, line_width=2)
        f.add_tools(HoverTool(renderers=[l], tooltips=[('Name', label), ]))
    f.legend.location = "top_right"
    f.legend.click_policy = "hide"
    f = add_hue_bar(f, length)
    bokeh.output_file(file_path+".html")
    bokeh.save(f)


def graphics_splom(labels: List[str], features: np.ndarray, file_path: str):
    """Generate graphic splom from features"""
    pdf_path = file_path.replace("XXX", "Splom")+".pdf"
    df = pd.DataFrame(data=features)
    column_packs = np.array_split(np.array(df.columns[1:]), math.ceil(len(df.columns[1:])/10))
    labels = [int(l) for l in labels]
    with PdfPages(pdf_path) as pdf:
        print("Gerando gráfico "+pdf_path)
        labels_set = set(labels)
        colors = list(mcolors.TABLEAU_COLORS.values())[: len(labels_set)]
        labels_color = np.array(labels, dtype=object)
        for index, l_s in enumerate(labels_set):
            labels_color[labels_color == l_s] = colors[index]
        progress_bar = others.ProgressBar("", len(column_packs), 0)
        for i, pack in enumerate(column_packs):
            progress_bar.print(i)
            plt.rcParams["figure.subplot.right"] = 0.8
            pd.plotting.scatter_matrix(df[pack], color=labels_color, alpha=.7, figsize=[8, 8])
            handles = [plt.plot([], [], ls="", color=c, alpha=.7, marker=".")[0] for c in colors]
            plt.legend(handles, list(dict.fromkeys(labels)), loc=(1.02, 0))
            pdf.savefig(bbox_inches='tight')
            plt.clf()
            plt.close()
        progress_bar.end()


def graphics_box2(classes: set, labels: list[str], features: np.ndarray, file_path: str):
    """Construct and save box chart to every class separate by feature"""
    pdf_path = file_path.replace("XXX", "BoxClassByFeatures")+".pdf"
    features_len = len(features[0])
    with PdfPages(pdf_path) as pdf:
        print("Construindo "+pdf_path)
        progress_bar = others.ProgressBar("", features_len, 0)
        for index in range(features_len):
            progress_bar.print(index)
            plt.clf()
            page_graphic = []
            for c in classes:
                positions = list(labels[:] == c)
                page_graphic.append(features[positions].T[index, :])
            plt.boxplot(page_graphic)
            plt.ylabel("Característica "+str(index))
            pdf.savefig(bbox_inches='tight')
        progress_bar.end()


def graphics_box1(classes: set, labels: list[str], features: np.ndarray, file_path: str):
    """Construct and save box chart to every feature separate by class"""
    _, axs = plt.subplots(len(classes), figsize=(len(features[0])/3, 10))
    for index, c in enumerate(classes):
        positions = list(labels[:] == c)
        axs[index].boxplot(features[positions])
        axs[index].set(ylabel="Notas de "+c)
    plt.savefig(file_path.replace("XXX", "BoxFeaturesByClass")+".pdf", bbox_inches='tight')


def graphics_lines(classes: set, labels: List[str],  features: list[list[int]], file_path: str, images_name: List[str]):
    legends = {"mean": [], "median": [], "mode": [], "Sandard Deviation": []}
    curves = {"mean": [], "median": [], "mode": [], "Sandard Deviation": []}
    for c in classes:
        positions = list(labels == c)
        for key in legends.keys():
            if key == "mean":
                curves[key].append(np.mean(features[positions], axis=0))
            elif key == "median":
                curves[key].append(np.median(features[positions], axis=0))
            elif key == "mode":
                curves[key].append(stats.mode(features[positions], axis=0, keepdims=True)[0][0])
            elif key == "Sandard Deviation":
                curves[key].append(np.std(features[positions], axis=0))
            legends[key].append("Nota de "+str(c))
        graphics_interactive(features[positions], np.array(images_name)[positions], file_path.replace("XXX", c))
    for key in legends.keys():
        graphics_interactive(curves[key], legends[key], file_path.replace("XXX", key))


def graphics_save(files_name: str, images_features: list):
    """Save graphics of the features to compare quality of distribution"""
    if len(images_features[0][1]) <= 500:
        features = []
        labels = np.array([])
        for item in images_features:
            labels = np.append(labels, item[0].split(".")[0])
            features.append(item[1])
        features = np.array(features)
        images_name = np.array(images_features, dtype=object)[:, 0]
        classes = set(labels)
        graphics_lines(classes, labels, features, files_name, images_name)
        if len(images_features[0][1]) <= 10:
            graphics_box1(classes, labels, features, files_name)
            graphics_box2(classes, labels, features, files_name)
            graphics_splom(labels, features, files_name)


def features_save(csv_name: str, images_features: list):
    """Save features in csv file"""
    os.makedirs(os.path.dirname(csv_name), exist_ok=True)
    with open(csv_name, "w") as csv_file:
        row = "Nome,Classe"
        csv_file.write(row)
        for image, feature in images_features:
            row = "\n"+str(image)+","+str(list(feature)).replace("[", "",).replace("]", "",)
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
