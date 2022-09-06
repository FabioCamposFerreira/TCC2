from PIL import Image


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
    """??

    Args:
        arq: string
            Path of the image

    Returns:
        : pil.image
            ??
    """
    im = Image.open(arq).convert(mode='HSV', palette=0)
    # gira imagem para ela ficar deitada
    l, h = im.size
    if l < h:
        im = im.rotate(angle=90, resample=0, expand=True)
    return im.resize((480,360), resample=Image.BICUBIC)
