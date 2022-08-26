# Um exemplo simples usado para validar os classificadores dentro do aplicativp
# recebe uma imagem pega o histograma e diz se esta claro ou escuro
import os
import time
import pickle
import image_processing
from sklearn.svm import SVC


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
    print(arq)
    classe = os.path.basename(arq).split(".")[0]
    print("| Processando imagem.")
    X = get_pattern(arq)
    y = classe
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

    Args:
        obj: 
            variable, functon or classe to save
        file: string
            name of the file to save

    Returns:
        None
            ??
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.closed

# grava um arquivo objeto SVC treinado


def trainer(X, y, method_name, method):
    """Trainer scikit-learn classifier usando patterns (X) and your classes (y)

    Args:
        X: array
            List of the patterns
        y: list
            Names of the classes
        method_name : str
            Name of the classifer
        method : class
            Scikit-leanr classifier
    Returns:
        classifier treined
            ??
    """
    print("Treinando "+method_name)
    st = time.time()
    method.fit(X, y)
    print("| Treinamento "
          + method_name
          + " durou "
          + str(time.time()-st)
          + "segundos")
    save_object(method, method_name+'.file')


# Configuring
methods = {
    "SVC": SVC(C=1.0,
               kernel='linear',
               degree=3, gamma='auto',
               coef0=0.0,
               shrinking=True,
               probability=True,
               tol=0.001,
               cache_size=200,
               class_weight=None,
               verbose=False,
               max_iter=-1,
               decision_function_shape='ovr',
               random_state=None),
}
data_base_path = "./../../Data_Base/Data_Base_Claro_Escuro/"
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
