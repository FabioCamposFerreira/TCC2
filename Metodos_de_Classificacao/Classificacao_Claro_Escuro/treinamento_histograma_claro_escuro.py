# Um exemplo simples usado para validar os classificadores dentro do aplicativp
# recebe uma imagem pega o histograma e diz se esta claro ou escuro
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from sklearn.svm import SVC
import pickle


def histogram(arq):
    """Gera o hitograma da imagem

    Args:
        arq: string
            Caminho da imagem

    Returns:
        : list
            O histograma da imagem
    """
    im = open_image(arq).getchannel(channel=2)
    return im.histogram(mask=None, extrema=None)


def open_image(arq):

    im = Image.open(arq).convert(mode='HSV', matrix=None,
                                 dither=None, palette=0, colors=256)
    # gira imagem para ela ficar deitada
    l, h = im.size
    if l < h:
        im = im.rotate(angle=90, resample=0, expand=True,
                       center=None, translate=None, fillcolor=None)
    return im.resize((5376, 3024), resample=Image.BICUBIC, box=None)

# Salva objetos em arquivos


def save_object(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.closed

# grava um arquivo objeto SVC treinado


def trainer_SVC(X, y, classification_folder):
    print("Treinando SVC")
    st = time.time()
    svc = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
              random_state=None)
    svc.fit(X, y)
    print("\t Treinando SVC durou "+str(time.time()-st)+"segundos")
    # salva SVC treinado nesta pasta e na pasta de classificação
    save_object(svc, 'SVC.file')
    save_object(svc, classification_folder+'SVC.file')


# Configarations
methods = (
    ("SVC", SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001,
                cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                random_state=None)),
)

# Configuring
data_base_path = "./../../Data_Base/Data_Base_Claro_Escuro/"
classes = [0, 1]  # 0:classe escura #1: classe clara
X = []
y = []

# Abrindo cada imagem do banco de dados para obter seu histograma da camada V gerando assim X e y
if __name__ == "__main__":
    for arq in os.listdir(data_base_path):
        print("Processando imagem "+arq)
        classe = arq.split(".")[0]
        h = histogram(data_base_path+arq)
        X.append(h)
        y = y + [classe]

    # treinando X e y
    # trainer_SVC(X, y)
    # trainer_RN(X, y)
    # trainer_KNN(X, y)
    # tempo total de treinamento
    print(time.perf_counter(), 'segundos')
