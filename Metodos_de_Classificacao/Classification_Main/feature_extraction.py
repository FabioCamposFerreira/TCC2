# Author: Fábio Campos Ferreira
# Contains modules for extracting features from the images processed by the **image_processing.py** modules
# In general, the modules receive the images processed by the **feature_extraction.py** modules, extracted and treated features with their identification labels, are the input to the **training.py** modules
# When run as **main**, graphics and tables are generated to compare the features

def histogram(im):
    """Receive pillow image and return histogram of the channel H"""
    return im.getchannel(channel=0).histogram(mask=None, extrema=None)


def get_features(im, feature):
    """Extract image features
    """
    if feature == "histogram":
        return histogram(im)
