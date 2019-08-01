import numpy as np
import torch
import skimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from segmentify.model import UNet, layers
from skimage import feature

def _load_model(pretrained_model):
    """Load the featurization model

    Parameters
    ----------
        model_path: str
            Path to the saved model file
    """
    # get pretraineod model saved filed path
    ## TODO better way to store and define file path
    if pretrained_model == "HPA":
        model_path = "/home/mars/CZI/segmentify/segmentify/model/saved_model/UNet_hpa_max.pth"
    elif pretrained_model == "nuclei":
        model_path = "/home/mars/CZI/segmentify/segmentify/model/saved_model/UNet_nuclei.pth"
    else:
        raise ValueError("pretrained model not defined")

    # load in saved model                                                     
    # TODO allow gpu
    pth = torch.load(model_path)
    model_args = pth['model_args']
    model_state = pth['model_state']
    model = UNet(**model_args)
    model.load_state_dict(model_state)

    # remove last layer and activation                                        
    model.segment = layers.Identity()
    model.activate = layers.Identity()
    model.eval()

    return model


def unet_featurize(image, pretrained_model="HPA"):
    """Featurize pixels in an image using pretrained UNet

    Parameters
    ----------
        image: numpy.ndarray
            Image data to be featurized
        pretrained_model: str (HPA)
            name of the pretraine model to use for featurization

    Returns
    -------
        features: np.ndarray
            One feature vector per pixel in the image
    """

    # TODO consider multiple images
    model = _load_model(pretrained_model)

    image = torch.Tensor(image).float()

    with torch.no_grad():
        features = model(image)

    features = features.squeeze().numpy()
    features = np.transpose(features, (1,2,0))
    return features


def filter_featurize(image):
    """Featurize pixels in an image using various image filters

    Parameters
    ----------
        image: np.ndarray
            Image data to be featurized

    Returns
    -------
        features: np.ndarray
            One feature vector per pixel in the image
    """
    # TODO consider multiple images
    image = np.squeeze(image)
    features = np.concatenate([[image],
                               [skimage.filters.gaussian(image,2)],
                               [skimage.filters.gaussian(image,4)],
                               [skimage.filters.sobel(image)],
                               [skimage.filters.laplace(image)],
                               [skimage.filters.gabor(image, 0.1)[0]],
                               [skimage.filters.gabor(image, 0.6)[1]],
                               [feature.canny(image, 1,0)]], axis=0)
    features = np.moveaxis(features, 0, -1)
    return features



def fit(image, labels, multichannel=False):
    """Train a pixel classifier.

    Parameters
    ----------
    image: np.ndarray
        Image data to be classified.
    labels: np.ndarray
        Sparse classification, where 0 pixels are ingored, and other integer
        values correspond to class membership.
    multichannel: bool, optional
        If image data is multichannel.

    Returns
    ----------
    classifier: sklearn.ensemble.RandomForestClassifier
        Object that can perform classifications
    """
    # pad input image
    w,h = image.shape
    w_padding = int((16-w%16)/2) if w%16 >0 else 0
    h_padding = int((16-h%16)/2) if h%16 >0 else 0
    image = np.pad(image, ((w_padding, w_padding),(h_padding, h_padding)), 'constant')

    clf = RandomForestClassifier(n_estimators=10)

    # TODO should this be elsewhere?
    while len(image.shape) < 4:
        image = np.expand_dims(image, 0)

    # TODO better way to choose featurizer
    features = unet_featurize(image)

    # crop out paddings
    if w_padding > 0:
        features = features[w_padding:-w_padding]
    if h_padding > 0:
        features = features[:,h_padding:-h_padding]

    X = features.reshape([-1, features.shape[-1]])
    y = labels.reshape(-1)
    X = X[y != 0]
    y = y[y != 0]

    if len(X) > 0:
        clf = clf.fit(X, y)

    return clf, features


def predict(classifier, features):
    """Train a pixel classifier.

    Parameters
    ----------
    classifier: sklearn.ensemble.RandomForestClassifier
        Object that can perform classifications
    features: np.ndarray
        featurized image
    multichannel: bool, optional
        If image data is multichannel.

    Returns
    ----------
    labels: np.ndarray
        Classification, where integer values correspond to class membership.
    """

    X = features.reshape([-1, features.shape[-1]])

    try:
        y = classifier.predict(X)
        labels = y.reshape(features.shape[:-1])
    except:
        # If classifer has not yet been fit return zeros
        labels = np.zeros(features.shape[:-1], dtype=int)
    return labels
