from PIL import Image



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