import numpy as np
import torch
import skimage
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from ..model import UNet, layers
from skimage import feature

def _load_model(featurizer_path):
    """Load the featurization model

    Parameters
    ----------
        featurizer_path: str
            Path to the saved model file

    Returns
    -------
    The loaded PyTorch model
    """

    # load in saved model                                                     
    pth = torch.load(featurizer_path)
    model_args = pth['model_args']
    model_state = pth['model_state']
    model = UNet(**model_args)
    model.load_state_dict(model_state)

    # remove last layer and activation                                        
    model.segment = layers.Identity()
    model.activate = layers.Identity()
    model.eval()

    return model


def unet_featurize(image, featurizer_path):
    """Featurize pixels in an image using pretrained UNet

    Parameters
    ----------
        image: numpy.ndarray
            Image data to be featurized
        featurizer_path: str (HPA)
            name of the pretraine model to use for featurization

    Returns
    -------
        features: np.ndarray
            One feature vector per pixel in the image
    """

    model = _load_model(featurizer_path)

    image = torch.Tensor(image).float()

    with torch.no_grad():
        features = model(image)

    features = features.numpy()
    features = np.transpose(features, (0,2,3,1))
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
    image = np.squeeze(image)
    features = np.concatenate([[image],
                               [skimage.filters.gaussian(image,2)],
                               [skimage.filters.gaussian(image,4)],
                               [skimage.filters.sobel(image)],
                               [skimage.filters.laplace(image)],
                               [skimage.filters.gabor(image, 0.6)[0]],
                               [skimage.filters.gabor(image, 0.6)[1]],
                               [feature.canny(image, 1,0)]], axis=0)
    features = np.moveaxis(features, 0, -1)
    return features



def fit(image, labels, featurizer="../model/saved_model/UNet_hpa_4c_mean_8.pth"):
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
    w,h = image.shape[-2:]

    w_padding = int((16-w%16)/2) if w%16 >0 else 0
    h_padding = int((16-h%16)/2) if h%16 >0 else 0

    padding = (0,) * (image.ndim - 2) + (w_padding, h_padding)
    padded_image = np.pad(image, (padding), 'constant')

    # make sure image has four dimentions (b,c,w,h)
    while len(padded_image.shape) < 4:
        padded_image = np.expand_dims(padded_image, 0)
    padded_image = np.transpose(padded_image, (1,0,2,3))

    # choose filter or unet featurizer
    if featurizer == "filter":
        padded_features = filter_featurize(padded_image)
    else:
        padded_features = unet_featurize(padded_image, featurizer)

    # crop out paddings
    if w_padding > 0:
        features = padded_features[:, w_padding:-w_padding]
    else:
        features = padded_features
    if h_padding > 0:
        features = features[:, :, h_padding:-h_padding]
    else:
        features = features

    # reshape and extract data
    X = features.reshape([-1, features.shape[-1]])
    y = labels.reshape(-1)
    X = X[y != 0]
    y = y[y != 0]

    # define and fit classifier
    clf = RandomForestClassifier(n_estimators=10)
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
        # get prediction and probability
        y = classifier.predict(X)
        prob = classifier.predict_proba(X)
        labels = y.reshape(features.shape[:-1])
        prob_shape = features.shape[:-1] + (prob.shape[-1], )
        prob = prob.reshape(prob_shape)

    except:
        # If classifer has not yet been fit return zeros
        labels = np.zeros(features.shape[:-1], dtype=int)
        prob = np.zeros(features.shape[:-1], dtype=int)
    return labels, prob
