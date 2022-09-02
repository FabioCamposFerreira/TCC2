from PIL import Image


def histogram(im):
    """Gera o hitograma da imagem

    Args:
        im : 
            pillow Image

    Returns:
        : list
            The histogram of the image
    """
    im = im.getchannel(channel=2)
    return im.histogram(mask=None, extrema=None)


def process_image(im):
    """
    Args: 
        im : 
            Pillow Image
    Returns: 
        : 
            Pillow Image processed
    """

    im = im.convert(mode='HSV', palette=0)
    # gira imagem para ela ficar deitada
    l, h = im.size
    if l < h:
        im = im.rotate(angle=90, resample=0, expand=True)
    im = im.resize((720, 576), resample=Image.BICUBIC)
    return im 


def process_texture(texture):
    """
    Args: 
        texture : 
            Object kivy.graphics.texture
    Returns: 
        : 
            Pillow Image processed
    """

    im = Image.frombytes('RGBA', texture.size, texture.pixels)
    return process_image(im)
