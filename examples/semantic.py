"""
Perform interactive semantic segmentation
"""

import numpy as np
from napari import Viewer
from napari.util import app_context
from skimage import data
from segmentify.semantic import fit, predict
import napari
print(napari.__version__)

coins = data.coins()
labels = np.zeros(coins.shape, dtype=int)

def segment(viewer):
    image = viewer.layers['input'].image
    labels = viewer.layers['train'].image
    clf = fit(image, labels)
    segmentation = predict(clf, image)
    print(np.unique(segmentation))
    viewer.layers['output'].image = segmentation

with app_context():

    # create an empty viewer
    viewer = Viewer()

    viewer.add_image(coins, name='input')
    viewer.layers['input'].colormap = 'gray'

    # add empty labels
    viewer.add_labels(labels, name='output')
    viewer.add_labels(labels, name='train')
    viewer.layers['train'].opacity = 0.9

    custom_key_bindings = {'s': segment}
    viewer.key_bindings = custom_key_bindings
