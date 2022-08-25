# Author: Fábio Campos
# Um exemplo simples usado para validar os classificadores dentro do aplicativp
# Recebe uma imagem pega o histograma e diz se esta claro ou escuro
import treinamento_histograma_claro_escuro as treining_codes
import os
import time
import pickle


def get_pattern(arq):
    """Extract image pattern

    Args:
        arq: string
            Path of the image

    Returns:
        : list
            The histogram of the image
    """
    return treining_codes.histogram(arq)


def make_sum_csv(row_last, colum_index, rows_len):
    if colum_index >= 25:
        colum_index -= 26
        row_last, colum_index = make_sum_csv(
            row_last, colum_index, rows_len
        )
    if colum[colum_index] == "Acertou?"\
            or colum[colum_index] == "Tempo de classificação":
        row_last += ";=SUM(" + chr(colum_index+65) +\
            str(3) + ":" + chr(colum_index+65) +\
            str(rows_len+2) + ")"+"/"+str(rows_len)
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
        : 
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
X = []
y = []
csv_file = open("resultados_para_cada_imagem.csv", "w")
line = "path_imagem, class, Clsf1 acertoos,Clsf1.predict,Clsf1.prob[0],Clsf1.prob[1], Clsf1 acertos,Clsf2.predict,Clsf2.prob[0],Clsf2.prob[1],...  "

# construc title line in csv_file
row_1 = "Imagem;Classe Correta"
row_2 = "-;-"
for method_name, method in methods:
    row_1 += ";" + method_name+";"+method_name
    row_2 += ";" + "Rotulo Gerado;Acertou?"
    for class_ in classes:
        row_1 += ";"+method_name
        row_2 += ";"+"Prob. da classe " + str(class_)
row_1 += ";"+method_name
row_2 += ";"+"Tempo de classificação"

csv_file.write(row_1+"\n")
csv_file.write(row_2+"\n")
# classifing data_base
for arq in os.listdir(data_base_path):
    row = ""
    print("Arquivo: "+arq)
    class_ = arq.split(".")[0]
    row = arq+";"+str(class_)
    print("| Processando imagem")
    pattern = get_pattern(data_base_path+arq)
    print("| Classificando")
    for method_name, method in methods:
        print("| | Rotulando usando "+method_name)
        row += to_rank(method_name, pattern, class_)
    print("| Salvando Resultdados")
    csv_file.write(row+"\n")
# make line sum in csv
colum = row_2.split(";")
rows_len = len(os.listdir(data_base_path))
row_last = ""
for colum_index in range(len(colum)):
    row_last, colum_index = make_sum_csv(row_last, colum_index, rows_len)
csv_file.write(row_last+"\n")
csv_file.close()
print(time.perf_counter(), 'segundos')
print("Classificação Concluida!")
