"""
Perform interactive semantic segmentation
"""

import numpy as np
from napari import Viewer
from napari.util import app_context
from skimage import data
from segmentify import semantic
from segmentify import instance


coins = data.coins()
labels = np.zeros(coins.shape, dtype=int)


def segment(viewer):
    image = viewer.layers['input'].image
    labels = viewer.layers['train'].image
    clf = semantic.fit(image, labels)
    segmentation = semantic.predict(clf, image)
    print('classes', len(np.unique(segmentation)))
    instances = instance.predict(image, segmentation)
    print('instances', instances.max())
    viewer.layers['classes'].image = segmentation
    viewer.layers['instances'].image = instances


with app_context():

    # create an empty viewer
    viewer = Viewer()

    viewer.add_image(coins, name='input')
    viewer.layers['input'].colormap = 'gray'

    # add empty labels
    viewer.add_labels(labels, name='classes')
    viewer.add_labels(labels, name='instances')
    viewer.add_labels(labels, name='train')
    viewer.layers['train'].opacity = 0.9

    custom_key_bindings = {'s': segment}
    viewer.key_bindings = custom_key_bindings
