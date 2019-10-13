"""Segmentify Viewer Class with key bindings
"""

import numpy as np
import math
import os
from . import util
from napari import Viewer as NapariViewer
from segmentify.semantic import fit, predict
from vispy.color import Colormap
from skimage import morphology, measure
from itertools import cycle
from scipy import stats

class Viewer(NapariViewer):
    """viewer for segmentify

    A NapariViewer based viewer with predefined keybindings for segmenting and post-processing

    Parameters
    ----------
    img : np.array
        Input image in numpy array format. If no image is passed in, matrix of zeros is used
    heatmap: bool
        If a probability heatmap layer should be created
    """

    def __init__(self, img=None):

        super(Viewer, self).__init__()

        # use empty image if none provided
        if img is None:
            self.img = np.zeros((256,256))
        else:
            self.img = img

        # class variables
        self.min_object_size = 25
        self.background_label = 1
        self.labels = np.zeros(self.img.shape, dtype=int)
        self.add_image(self.img, name='input')
        self.selem = morphology.selem.square(3)
        self.prob = None
        self.segmentation = None

        if len(img.shape) > 2:
            self.selem = np.array([self.selem])

        # create empty heatmap label
        self.probability_heatmap = self.add_image(self.labels.astype(float), \
                                                  name="prediction probability")
        self.probability_heatmap.opacity = 0.0
        self.colormap = Colormap([[0.0,0.0,0.0,0.0],[1.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0]])

        # add label layers
        self.add_labels(self.labels, name='output')
        self.add_labels(self.labels, name='train')
        self.layers['train'].opacity = 0.9

        # define featurizers
        featurizer_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"model","saved_model")
        featurizer_paths = os.listdir(featurizer_dir)
        featurizer_paths = sorted([os.path.join(featurizer_dir,path) for path in featurizer_paths])
        featurizer_paths.append("filter")
        self.featurizers = cycle(featurizer_paths)
        self.cur_featurizer = next(self.featurizers)
        self.status = self.cur_featurizer.split("/")[-1]

        # key-bindings
        self.bind_key('Shift-S', self.segment)
        self.bind_key('Shift-H', self.show_heatmap)
        self.bind_key('Shift-N', self.next_featurizer)
        self.bind_key('Shift-D', self.dilation)
        self.bind_key('Shift-E', self.erosion)
        self.bind_key('Shift-C', self.closing)
        self.bind_key('Shift-O', self.opening)
        self.bind_key('Shift-F', self.fill_holes)

    def _extract_label(func):
        """decorator that extract pixels with selected label

        This decorator takes in a function with a Napari Viewer parameter,
        and only run the function on pixels with selected labels.
        The output of the function will replace the original selected pixels.


        Parameters
        ----------
        func : python function
            Function to be decorated

        Returns
        -------
        new_func: python function
            The decorated function
        """

        def new_func(self, viewer):
            """Decorator function that extracts selected label

            Parameters
            ----------
            viewer : Segmentify Viewer
            """

            curr_label = viewer.active_layer.selected_label

            # extract only pixels with selected label
            original = viewer.active_layer.data
            labeled_img = np.zeros_like(original)
            labeled_img[original == curr_label] = 1
            viewer.active_layer.data = labeled_img

            # run morphology
            viewer.active_layer.data = func(self, viewer)

            # merge processed image with original
            all_labels = np.unique(original)
            if len(all_labels) == 2:
                background_label = all_labels[all_labels != curr_label][0]
                original[(viewer.active_layer.data == 0) & (original == curr_label)] = background_label
            else:
                original[(viewer.active_layer.data == 0) & (original == curr_label)] = self.background_label
            original[viewer.active_layer.data == 1] = curr_label

            viewer.active_layer.data = original

            viewer.status = f"Finished {func.__name__} on label {viewer.active_layer.selected_label}"

        return new_func


    def segment(self, viewer):
        """function that fit and segment input image (key-binding: SHIFT-S)

        This function takes in a Segmentify Viewer and featurize the input image.
        A Random Forest Classifier is trained based on the selected training labels.
        The entire input image is segmented based on the Random Forest Classifier's predictions.

        Parameters
        ----------
        viewer : Segmentify Viewer
        """

        viewer.status = "Segmenting..."

        # get data
        image = viewer.layers['input'].data
        labels = viewer.layers['train'].data

        # fit and predict
        clf, features = fit(image, labels, featurizer=self.cur_featurizer)
        segmentation, self.prob = predict(clf, features)

        # show prediction
        self.segmentation = np.squeeze(segmentation)
        viewer.layers['output'].data = self.segmentation

        viewer.status = "Segmentation Completed"

    def show_heatmap(self, viewer):
        """This function generates the confidence heatmap of model's prediction.

        The heatmap is generated based on the normalized entropy of the prediction probabilities.
        """

        if self.prob is not None:
            # calcualte entropy for probability
            prob = np.apply_along_axis(util._norm_entropy,-1, self.prob)
            prob = (prob - np.min(prob)) / np.ptp(prob)
            prob = np.squeeze(prob)

            self.probability_heatmap.colormap = "uncertainty", self.colormap
            self.probability_heatmap.data = prob
            self.probability_heatmap.opacity = 0.7
            viewer.status = "Probability heatmap generated"
        else:
            viewer.status = "Segmentation required before heatmap generation"

    def next_featurizer(self, viewer):
        """get next featurizer from self.featurizer (key-binding: SHIFT-N)

        This function cycles through the availible featurizers in segmentify/model/saved_model,
        as well as the image filter featurizer.

        Parameters
        ----------
        viewer : Segmentify Viewer
        """
        self.cur_featurizer = next(self.featurizers)
        viewer.status = self.cur_featurizer.split("/")[-1]


    def closing(self, viewer):
        """apply the closing operation (key-binding: SHIFT-C)

        This function applies the closing operation by dilating the selected label pixels,
        following by erosion

        Parameters
        ----------
        viewer : Segmentify Viewer

        Returns
        -------
        The procssed image
        """
        viewer.status = "Closing"
        _ = self.dilation(viewer)
        processed_img = self.erosion(viewer)
        viewer.status = f"Finished closing on label {viewer.active_layer.selected_label}"
        return processed_img

    def opening(self, viewer):
        """apply the opening operation (key-binding: SHIFT-O)

        This function applies the opening operation by eroding the selected label pixels,
        following by dilation

        Parameters
        ----------
        viewer : Segmentify Viewer

        Returns
        -------
        The procssed image
        """
        viewer.status = "Closing"
        _ = self.erosion(viewer)
        processed_img = self.dilation(viewer)
        viewer.status = f"Finished opening on label {viewer.active_layer.selected_label}"
        return processed_img

    def erosion(self, viewer):
        """apply the erosion operation (key-binding: SHIFT-E)

        This function applies the erosion operation on selected label pixels

        Parameters
        ----------
        viewer : Segmentify Viewer

        Returns
        -------
        The procssed image
        """
        viewer.status = "Eroding"
        processed_img = util._erode_img(viewer.active_layer.data, \
                                                      target_label=viewer.active_layer.selected_label)
        viewer.active_layer.data = processed_img
        viewer.status = f"Finished erosion on label {viewer.active_layer.selected_label}"
        return processed_img

    @_extract_label
    def dilation(self, viewer):
        """apply the dilation operation (key-binding: SHIFT-D)

        This function applies the dilation operation on selected label pixels

        Parameters
        ----------
        viewer : Segmentify Viewer

        Returns
        -------
        The procssed image
        """
        processed_img =  morphology.dilation(viewer.active_layer.data, self.selem)
        viewer.status = f"Finished Dilation on label {viewer.active_layer.selected_label}"
        return processed_img

    @_extract_label
    def fill_holes(self, viewer):
        """apply the fill holes operation (key-binding: SHIFT-D)

        This function applies the fill holes operation on the selected label pixels

        Parameters
        ----------
        viewer : Segmentify Viewer

        Returns
        -------
        The procssed image
        """
        if len(viewer.active_layer.data.shape) > 2:
            processed_imgs = []
            for i in range(viewer.active_layer.data.shape[0]):
                processed_img = morphology.remove_small_holes(viewer.active_layer.data[i].astype(bool)).astype(int)
                processed_imgs.append(processed_img)
            return np.stack(processed_imgs, 0)
        else:
            return morphology.remove_small_holes(viewer.active_layer.data.astype(bool)).astype(int)

