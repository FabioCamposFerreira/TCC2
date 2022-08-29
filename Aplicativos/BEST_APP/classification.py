import image_processing
import pickle


def classifiy(im):
    """Extract image pattern

    Args:
        im : 
            pillow Image

    Returns:
        y : int or None
            The class labed or None
    """
    classifier = "SVC"
    pattern = get_pattern(im)
    clsf = read_object(classifier+".file")
    predct_proba = clsf.predict_proba([pattern])
    id_max = predct_proba[0].argmax()
    # print("\033[91m {}\033[00m" .format(id_max))
    # print("\033[92m {}\033[00m" .format(predct_proba[0][id_max]))
    # analyze criteria for sure
    if predct_proba[0][id_max] > .7:
        y = id_max
    else:
        y = None
    return y


def get_pattern(im):
    """Extract image pattern

    Args:
        im: 
            pillow Image

    Returns:
        : list
            The patterns
    """
    return image_processing.histogram(im)


def read_object(arq):
    """Le objeto no arquivo

    Args:
        arq: string
            localização do arquivo

    Returns:
        dump : object
            O objeto contido no arquivo
    """
    with open(arq, "rb") as f:
        dump = pickle.load(f)
        return dump
