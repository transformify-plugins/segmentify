"""
Perform interactive semantic segmentation on 3D volume
"""
import numpy as np
import napari
import imageio
import os
from napari import Viewer, gui_qt
from skimage import data, io
from skimage.color import rgb2gray
from segmentify.semantic import fit, predict

print(napari.__version__)


with gui_qt():

    # create an empty viewer
    viewer = Viewer()

    # read in sample data
    example_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),"image_stack.tif")
    nuclei = io.imread(example_file)
    #nuclei = np.transpose(nuclei, (2,0,1))
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

        clf, features = fit(image, labels)

        segmentation = predict(clf, features)
        segmentation = np.squeeze(segmentation)
        viewer.layers['output'].data = segmentation
