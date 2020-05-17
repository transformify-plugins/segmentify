"""Segmentify Viewer Class with key bindings
"""

import numpy as np
from magicgui import magicgui
from napari import layers
from .semantic import fit, predict
from .util import Featurizers, norm_entropy


@magicgui(call_button="execute")
def segmentation(base_image: layers.Image,
                       initial_labels: layers.Labels,
                       featurizer:Featurizers) -> layers.Labels:
    """Segment an image based on an initial labeling."""

    print("Segmenting...")

    # fit and predict
    if base_image.rgb:
        data = base_image.data.mean(axis=2)
    else:
        data = base_image.data

    # normalize image based on clims
    clims = base_image.contrast_limits
    data = (data - clims[0]) / (clims[1] - clims[0])

    clf, features = fit(data, initial_labels.data, featurizer=featurizer.value)
    image_features = np.squeeze(features).transpose(2, 0, 1)
    segmentation, probability = predict(clf, features)

    # calcualte entropy from probability
    entropy = np.apply_along_axis(norm_entropy, -1, probability)
    entropy = (entropy - np.min(entropy)) / np.ptp(entropy)
    entropy = np.squeeze(entropy)

    print("... Completed")  

    return np.squeeze(segmentation)
