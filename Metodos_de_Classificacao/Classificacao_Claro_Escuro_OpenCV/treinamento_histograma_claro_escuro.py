# Um exemplo simples usado para validar os classificadores dentro do aplicativp
# recebe uma imagem pega o histograma e diz se esta claro ou escuro
import os
import time
import pickle
import image_processing
import cv2 as cv
import numpy as np

def create_X_y(arq):
    """Return class and feature of the imagem

    Args:
        arq: string
            Path of the image

    Returns:
        X: list
            Features of the image
        y: int
            classe of the feature
    """
    classe = os.path.basename(arq).split(".")[0]
    print("| Processando imagem "+str(os.path.basename(arq)))
    X = get_pattern(arq)
    y = [int(classe)]
    return X, y


def get_pattern(arq):
    """Extract image pattern

    Args:
        arq: string
            Path of the image

    Returns:
        : list
            The histogram of the image
    """
    return image_processing.histogram(arq)


# Salva objetos em arquivos


def save_object(obj, file):
    """??
    """
    obj.save(file)

# grava um arquivo objeto SVC treinado


def trainer(X, y, method_name, method):

    print("Treinando "+method_name)
    st = time.time()
    X = np.matrix(X, dtype=np.float32)
    y = np.array(y)
    method.train(X, cv.ml.ROW_SAMPLE, y)
    print("| Treinamento "
          + method_name
          + " durou "
          + str(time.time()-st)
          + "segundos")
    save_object(method, method_name+'.file')


# Configuring
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
methods = {
    "SVM": svm,
}
data_base_path = "../../Data_Base/Data_Base_Claro_Escuro/"
classes = [0, 1]  # 0:classe escura #1: classe clara
X = []
y = []

if __name__ == "__main__":
    data_base = os.listdir(data_base_path)
    for arq in os.listdir(data_base_path):
        cXy = create_X_y(data_base_path+arq)
        X += [cXy[0]]
        y += cXy[1]
    # print("\033[91m {}\033[00m" .format(y))
    for method in methods:
        trainer(X, y, method, methods[method])
    # Total run time
    print(time.perf_counter(), 'segundos')
