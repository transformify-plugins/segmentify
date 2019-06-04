import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from skimage.filters import gaussian

def featurize(image, multichannel=False):
    """Featurize pixels in an image.

    Parameters
    ----------
	image: np.ndarray
		Image data to be featurized.
	multichannel: bool, optional
		If image data is multichannel.

    Returns
    ----------
    features: np.ndarray
        One feature vector per pixel in the image.
    """

    if not multichannel:
        features = np.concatenate([[image], [gaussian(image, 2)],
                                   [gaussian(image, 4)]], axis=0)
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

    clf = RandomForestClassifier(n_estimators=10)

    features = featurize(image, multichannel=multichannel)

    X = features.reshape([-1, features.shape[-1]])
    y = labels.reshape(-1)
    X = X[y != 0]
    y = y[y != 0]

    if len(X) > 0:
        clf = clf.fit(X, y)

    return clf


def predict(classifier, image, multichannel=False):
    """Train a pixel classifier.

    Parameters
    ----------
    classifier: sklearn.ensemble.RandomForestClassifier
        Object that can perform classifications
    image: np.ndarray
		Image data to be classified.
	multichannel: bool, optional
		If image data is multichannel.

    Returns
    ----------
    labels: np.ndarray
        Classification, where integer values correspond to class membership.
    """

    features = featurize(image, multichannel=multichannel)

    X = features.reshape([-1, features.shape[-1]])

    try:
        y = classifier.predict(X)
        labels = y.reshape(image.shape)
    except:
        # If classifer has not yet been fit return zeros
        labels = np.zeros(image.shape, dtype=int)

    return labels
