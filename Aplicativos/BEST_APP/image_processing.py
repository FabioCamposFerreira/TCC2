from PIL import Image


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
    # rotate to lay down the image
    l, h = im.size
    if l < h:
        im = im.rotate(angle=90, resample=0, expand=True)
    im = im.resize((854, 480), resample=Image.NEAREST)
    # im.convert("RGB").save("".join(("Imagem Processada", ".png")))
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
    # im = Image.open("10.32.png")
    # im.convert("RGB").save("".join(("Imagem Capturada", ".png")))
    return process_image(im)
