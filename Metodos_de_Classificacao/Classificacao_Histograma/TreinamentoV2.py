import glob
from typing import Any, Union, Tuple, Iterable
from PIL import Image
from matplotlib.pyplot import hist
from numpy.core.multiarray import ndarray
from sklearn.svm import SVC
import pickle
import time
from sklearn.neural_network import MLPClassifier
import sklearn.neighbors as skl
import matplotlib.pyplot as plot
import numpy as np


# abre image e faz o tratamento e padronização
def open_image(arq):
    im = Image.open(arq).convert(mode='L', matrix=None, dither=None, palette=0, colors=256)
    l, h = im.size
    if l < h:
        im = im.rotate(angle=90, resample=0, expand=True, center=None, translate=None, fillcolor=None)
    l, h = im.size
    h_cut = int((h / 2) * 0.1)
    l_cut = int((l / 2) * 0.1)
    im = im.crop(box=(l_cut, h_cut, l - l_cut, h - h_cut))
    im = im.resize((5376, 3024), resample=Image.BICUBIC, box=None)
    return im

# recebe a imagem e retorna o vetor de caracteristicas de cor da imagem


def histogram(file):
    im = open_image(arq=file)
    im = im.getchannel(channel=0)
    h = im.histogram(mask=None, extrema=None)
    return h


# grava um arquivo objeto SVC treinado
def trainer_SVC(X, y):
    svc = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
              random_state=None)
    svc.fit(X, y)
    save_object(svc, 'SVC.file')


# grava um arquivo objeto RN treinado
def trainer_RN(X, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    clf.fit(X, y)
    save_object(clf, 'RN.file')


# grava um arquivo objeto KNN treinado
def trainer_KNN(X, y):
    KNN = skl.KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    KNN.fit(X, y)
    save_object(KNN, 'KNN.file')


# Salva objetos em arquivos
def save_object(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.closed


# Abrindo cada imagem do banco de dados para obter seu histograma da camada H gerando assim X e y
classes = [2, 5, 10, 20, 50, 100]
X = []
y = []
for classe in classes:
    for arq in sorted(glob.glob("/home/suporte/Documents/TCC2/TCC2/Data Base/DATA 1/*")):
        name = arq[len('/home/suporte/Documents/TCC2/TCC2/Data Base/DATA 1/'):(len(arq) - 4)]
        name_int = np.floor(float(name))
        if name_int == classe:
            h = histogram(arq)
            X.append(h)
            y = y + [classe]


# treinando X e y
trainer_SVC(X, y)
trainer_RN(X, y)
trainer_KNN(X, y)

# tempo total de treinamento
print(time.perf_counter(), 'segundos')