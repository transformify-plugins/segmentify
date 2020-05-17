"""Segmentify Viewer Class with key bindings
"""

import numpy as np
import math
import os
from . import util
from .semantic import fit, predict
from .util import Featurizers
from magicgui import magicgui
from skimage import morphology, measure
from itertools import cycle
from scipy import stats
from napari import layers


@magicgui(call_button="execute")
def segmentation(base_image: layers.Image,
                       initial_labels: layers.Labels,
                       featurizer:Featurizers) -> layers.Labels:
    """Segment an image based on an initial labeling."""

    print("Segmenting...")

    # fit and predict
    clf, features = fit(base_image.data, initial_labels.data, featurizer=featurizer.value)
    image_features = features[0].transpose(2, 0, 1)
    segmentation, probability = predict(clf, features)

    print("... Completed")  

    return np.squeeze(segmentation)


    # # key-bindings
    # self.bind_key('Shift-S', self.segment)
    # self.bind_key('Shift-H', self.show_heatmap)
    # self.bind_key('Shift-D', self.dilation)
    # self.bind_key('Shift-E', self.erosion)
    # self.bind_key('Shift-C', self.closing)
    # self.bind_key('Shift-O', self.opening)
    # self.bind_key('Shift-F', self.fill_holes)

    # def _extract_label(func):
    #     """decorator that extract pixels with selected label
    # 
    #     This decorator takes in a function with a Napari Viewer parameter,
    #     and only run the function on pixels with selected labels.
    #     The output of the function will replace the original selected pixels.
    # 
    # 
    #     Parameters
    #     ----------
    #     func : python function
    #         Function to be decorated
    # 
    #     Returns
    #     -------
    #     new_func: python function
    #         The decorated function
    #     """
    # 
    #     def new_func(self, viewer):
    #         """Decorator function that extracts selected label
    # 
    #         Parameters
    #         ----------
    #         viewer : Segmentify Viewer
    #         """
    # 
    #         curr_label = viewer.active_layer.selected_label
    # 
    #         # extract only pixels with selected label
    #         original = viewer.active_layer.data
    #         labeled_img = np.zeros_like(original)
    #         labeled_img[original == curr_label] = 1
    #         viewer.active_layer.data = labeled_img
    # 
    #         # run morphology
    #         viewer.active_layer.data = func(self, viewer)
    # 
    #         # merge processed image with original
    #         all_labels = np.unique(original)
    #         if len(all_labels) == 2:
    #             background_label = all_labels[all_labels != curr_label][0]
    #             original[(viewer.active_layer.data == 0) & (original == curr_label)] = background_label
    #         else:
    #             original[(viewer.active_layer.data == 0) & (original == curr_label)] = self.background_label
    #         original[viewer.active_layer.data == 1] = curr_label
    # 
    #         viewer.active_layer.data = original
    # 
    #         viewer.status = f"Finished {func.__name__} on label {viewer.active_layer.selected_label}"
    # 
    #     return new_func


    # def show_heatmap(self, viewer):
    #     """This function generates the confidence heatmap of model's prediction.
    # 
    #     The heatmap is generated based on the normalized entropy of the prediction probabilities.
    #     """
    # 
    #     if self.prob is not None:
    #         # calcualte entropy for probability
    #         prob = np.apply_along_axis(util._norm_entropy,-1, self.prob)
    #         prob = (prob - np.min(prob)) / np.ptp(prob)
    #         prob = np.squeeze(prob)
    # 
    #         self.probability_heatmap.colormap = "uncertainty", self.colormap
    #         self.probability_heatmap.data = prob
    #         self.probability_heatmap.opacity = 0.7
    #         viewer.status = "Probability heatmap generated"
    #     else:
    #         viewer.status = "Segmentation required before heatmap generation"



    # def closing(self, viewer):
    #     """apply the closing operation (key-binding: SHIFT-C)
    # 
    #     This function applies the closing operation by dilating the selected label pixels,
    #     following by erosion
    # 
    #     Parameters
    #     ----------
    #     viewer : Segmentify Viewer
    # 
    #     Returns
    #     -------
    #     The procssed image
    #     """
    #     viewer.status = "Closing"
    #     _ = self.dilation(viewer)
    #     processed_img = self.erosion(viewer)
    #     viewer.status = f"Finished closing on label {viewer.active_layer.selected_label}"
    #     return processed_img
    # 
    # def opening(self, viewer):
    #     """apply the opening operation (key-binding: SHIFT-O)
    # 
    #     This function applies the opening operation by eroding the selected label pixels,
    #     following by dilation
    # 
    #     Parameters
    #     ----------
    #     viewer : Segmentify Viewer
    # 
    #     Returns
    #     -------
    #     The procssed image
    #     """
    #     viewer.status = "Closing"
    #     _ = self.erosion(viewer)
    #     processed_img = self.dilation(viewer)
    #     viewer.status = f"Finished opening on label {viewer.active_layer.selected_label}"
    #     return processed_img
    # 
    # def erosion(self, viewer):
    #     """apply the erosion operation (key-binding: SHIFT-E)
    # 
    #     This function applies the erosion operation on selected label pixels
    # 
    #     Parameters
    #     ----------
    #     viewer : Segmentify Viewer
    # 
    #     Returns
    #     -------
    #     The procssed image
    #     """
    #     viewer.status = "Eroding"
    #     processed_img = util._erode_img(viewer.active_layer.data, \
    #                                                   target_label=viewer.active_layer.selected_label)
    #     viewer.active_layer.data = processed_img
    #     viewer.status = f"Finished erosion on label {viewer.active_layer.selected_label}"
    #     return processed_img
    # 
    # @_extract_label
    # def dilation(self, viewer):
    #     """apply the dilation operation (key-binding: SHIFT-D)
    # 
    #     This function applies the dilation operation on selected label pixels
    # 
    #     Parameters
    #     ----------
    #     viewer : Segmentify Viewer
    # 
    #     Returns
    #     -------
    #     The procssed image
    #     """
    #     processed_img =  morphology.dilation(viewer.active_layer.data, self.selem)
    #     viewer.status = f"Finished Dilation on label {viewer.active_layer.selected_label}"
    #     return processed_img
    # 
    # @_extract_label
    # def fill_holes(self, viewer):
    #     """apply the fill holes operation (key-binding: SHIFT-D)
    # 
    #     This function applies the fill holes operation on the selected label pixels
    # 
    #     Parameters
    #     ----------
    #     viewer : Segmentify Viewer
    # 
    #     Returns
    #     -------
    #     The procssed image
    #     """
    #     if len(viewer.active_layer.data.shape) > 2:
    #         processed_imgs = []
    #         for i in range(viewer.active_layer.data.shape[0]):
    #             processed_img = morphology.remove_small_holes(viewer.active_layer.data[i].astype(bool)).astype(int)
    #             processed_imgs.append(processed_img)
    #         return np.stack(processed_imgs, 0)
    #     else:
    #         return morphology.remove_small_holes(viewer.active_layer.data.astype(bool)).astype(int)
