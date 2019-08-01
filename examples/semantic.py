"""
Perform interactive semantic segmentation
"""
import numpy as np
from napari import Viewer, gui_qt
from skimage import data
from skimage.color import rgb2gray
from segmentify.semantic import fit, predict
import napari
import imageio
import os
print(napari.__version__)



with gui_qt():

    # create an empty viewer
    viewer = Viewer()

    # read in sample data
    example_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "nuclei.png")
    nuclei = imageio.imread(example_file)
    nuclei = rgb2gray(nuclei)
    labels = np.zeros(nuclei.shape, dtype=int)

    viewer.add_image(nuclei, name='input')

    # add empty labels
    viewer.add_labels(labels, name='output')
    viewer.add_labels(labels, name='train')
    viewer.layers['train'].opacity = 0.9

    @viewer.bind_key('s')
    def segment(viewer):
        image = viewer.layers['input'].data
        labels = viewer.layers['train'].data

        clf = fit(image, labels)

        segmentation = predict(clf, image)
        segmentation = np.squeeze(segmentation)
        viewer.layers['output'].data = segmentation
