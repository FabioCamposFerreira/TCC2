# Um exemplo simples usado para validar os classificadores dentro do aplicativp
# recebe uma imagem pega o histograma e diz se esta claro ou escuro
from PIL import Image
import os
import time
import pickle

# gera o histograma de um arquivo imagem
def histogram(arq):
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

# Abrindo cada imagem do banco de dados para obter seu histograma da camada V gerando assim X e y
data_base_path = "./../../Data_Base/Data_Base_Claro_Escuro/"
classes = [0, 1]  # 0:classe escura #1: classe clara
X = []
y = []
for arq in os.listdir(data_base_path):
    print("Processando imagem "+arq)
    classe = arq.split(".")[0]
    h = histogram(data_base_path+arq)

# tempo total de classificacao
print(time.perf_counter(), 'segundos')
