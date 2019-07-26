"""
Perform interactive semantic segmentation
"""

import numpy as np
from napari import Viewer, gui_qt
from skimage import data
from segmentify import semantic
from segmentify import instances


coins = data.coins()
labels = np.zeros(coins.shape, dtype=int)


with gui_qt():

    # create an empty viewer
    viewer = Viewer()

    viewer.add_image(coins, name='input', colormap='gray')

    # add empty labels
    viewer.add_labels(labels, name='classes')
    viewer.add_labels(labels, name='instances')
    viewer.add_labels(labels, name='train')
    viewer.layers['train'].opacity = 0.9

    @viewer.bind_key('s')
    def segment(viewer):
        image = viewer.layers['input'].data
        labels = viewer.layers['train'].data
        clf = semantic.fit(image, labels)
        segmentation = semantic.predict(clf, image)
        print('classes', len(np.unique(segmentation)))
        instances = instances.predict(image, segmentation)
        print('instances', instances.max())
        viewer.layers['classes'].data = segmentation
        viewer.layers['instances'].data = instances
