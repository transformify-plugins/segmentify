import numpy as np
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects


def predict(image, labels, multichannel=False):
    """Segments objects in an image into instances using background,
    foreground, and boundary information in labels.

    Parameters
    ----------
	image: np.ndarray
		Image data to be featurized.
    labels: np.ndarray
        Semanitc classification of image into background '1', foreground '2',
        and boundary '3' pixels.
	multichannel: bool, optional
		If image data is multichannel.

    Returns
    ----------
    instances: np.ndarray
        Integer labels corresponding to each object
    """

    # extract foreground
    bw = closing(labels == 2, square(10))

    # remove borders
    bw = bw & ~(labels == 3)

    # remove small objects
    cleared = remove_small_objects(bw, 40)

    # label image regions
    instances = label(cleared)

    return instances
