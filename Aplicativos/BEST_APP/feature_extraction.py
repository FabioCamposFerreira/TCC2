def histogram(im):
    """Receive image and return histogram of the channel H"""
    return im.getchannel(channel=0).histogram(mask=None, extrema=None)


def normalize(list_):
    """Normalize list or array"""
    x_max = max(list_)
    x_min = min(list_)
    difference = x_max-x_min
    if not difference:
        raise Exception("Extract feature is a string of zeros")
    return [(x-x_min)/(difference) for x in list_]


def get_pattern(im):
    return (histogram(im))
