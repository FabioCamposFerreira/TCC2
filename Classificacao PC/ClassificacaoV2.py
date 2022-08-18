from PIL import Image
import pickle
import time
import glob
import numpy as np



# abre image e aplica filtros para uma melhor classificação
def open_image(arq):
    im = Image.open(arq).convert(mode='L', matrix=None, dither=None, palette=0, colors=256)
    l, h = im.size
    if l < h:
        im = im.rotate(angle=90, resample=0, expand=True, center=None, translate=None, fillcolor=None)
    l, h = im.size
    h_cut = int((h / 2) * 0.1)
    l_cut = int((l / 2) * 0.1)
    im = im.crop(box=(l_cut, h_cut, l - l_cut, h - h_cut))
    im = im.resize((5376 , 3024), resample=Image.BICUBIC, box=None)
    return im


# ler objeto no arquivo
def read_object(file):
    with open(file, "rb") as f:
        dump = pickle.load(f)
        return dump


# recebe a imagem eretorna o vetor de caracteristicas de cor da imagem
def histogram(file):
    im = open_image(arq=file)
    im = im.getchannel(channel=0)
    h = im.histogram(mask=None, extrema=None)
    return h

# retorna nova imagem classificada
def regimentation(arq, obj_file):
    point = histogram(file=arq)
    clsf = read_object(obj_file)
    return clsf.predict_proba([point])

#recebe a pasta a ser classificadoa e o classificador gera um arquivo texto usando os classificadores com dados (nome da imagem, classificador, probabilidades e acerto ou erro) para transformar em csv
def result(pasta, classifier):
    temp = open(classifier+'.csv', 'w')
    c= 'Treinador;Pasta;Imagem;Classificação;Pontuação;Nota de 2;Nota de 5;Nota de 10;Nota 20;Nota 50;Nota 100;Acuracia\n'
    temp.write(c)

    classes = [2, 5, 10, 20, 50, 100]
    for arq in sorted(glob.glob("C:/DATA/"+pasta+"/*")):
        name = arq[len('C:/DATA/'+pasta+'/'):(len(arq) - 4)]
        name_int = int(float(name[1:]))
        prob = regimentation(arq,classifier+'.file')[0]
        a = sorted(prob)
        j=0
        for i in prob:
            if i == a[-1]:
                position = j
            j += 1
        classe = classes[position]
        if name_int== classe or classe==(name_int-1):
            result = 1
        else:
            result=0

        c = classifier+';'+pasta+';'+name+';'+str(classe)+';'+str(result)+';'+str(prob[0])+';'+str(prob[1])+';'+str(prob[2])+';'+str(prob[3])+';'+str(prob[4])+';'+str(prob[5])+'\n'
        temp.write(c)
    
    c = ';;;;=SOMA(E2:E32)\n'
    temp.write(c)
    temp.close()

#executa a classificacao para cada tipo de classificador
result('SVC')
result('KNN')
result('RN')

print(time.perf_counter(), 'segundos')
