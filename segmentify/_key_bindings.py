from napari.layers.labels import Labels
import math
import os
from skimage import morphology, measure
from scipy import stats
from .util import erode_img


@Labels.bind_key('Shift-C')
def closing(self, layer):
    """Apply the closing operation (key-binding: SHIFT-C)

    This function applies the closing operation by dilating the selected label
    pixels, following by erosion

    Parameters
    ----------
    layer : napari.layers.Labels
    """
    dilation(layer)
    erosion(layer)


@Labels.bind_key('Shift-O')
def opening(self, layer):
    """Apply the opening operation (key-binding: SHIFT-O)

    This function applies the opening operation by eroding the selected label
    pixels, following by dilation

    Parameters
    ----------
    layer : napari.layers.Labels
    """
    erosion(layer)
    dilation(layer)


@Labels.bind_key('Shift-E')
def erosion(self, layer):
    """Apply the erosion operation (key-binding: SHIFT-E)

    This function applies the erosion operation on selected label pixels

    Parameters
    ----------
    layer : napari.layers.Labels
    """
    labeled = extract_label(layer.data, layer.selected_label)
    selem = morphology.selem.square(3)
    processed_img = erode_img(layer.data, target_label=layer.selected_label)
    merged = merge_label(processed_img, layer.data, layer.selected_label)
    layer.data = merged


@Labels.bind_key('Shift-D')
def dilation(self, layer):
    """Apply the dilation operation (key-binding: SHIFT-D)

    This function applies the dilation operation on selected label pixels

    Parameters
    ----------
    layer : napari.layers.Labels
    """
    labeled = extract_label(layer.data, layer.selected_label)
    selem = morphology.selem.square(3)
    processed_img =  morphology.dilation(labeled, selem)
    merged = merge_label(processed_img, layer.data, layer.selected_label)
    layer.data = merged


@Labels.bind_key('Shift-F')
def fill_holes(self, layer):
    """apply the fill holes operation (key-binding: SHIFT-D)

    This function applies the fill holes operation on the selected label pixels

    Parameters
    ----------
    viewer : Segmentify Viewer

    Returns
    -------
    The procssed image
    """
    labeled = extract_label(layer.data, layer.selected_label)
    if len(labeled.shape) > 2:
        processed_imgs = []
        for i in range(labeled.shape[0]):
            processed_img = morphology.remove_small_holes(labeled[i].astype(bool)).astype(int)
            processed_imgs.append(processed_img)
        processed_img = np.stack(processed_imgs, 0)
    else:
        processed_img = morphology.remove_small_holes(labeled.astype(bool)).astype(int)
    merged = merge_label(processed_img, layer.data, layer.selected_label)
    layer.data = merged


def extract_label(data, label):
    """Extract data pixels with selected label"""
    labeled = np.zeros_like(data)
    labeled[data == label] = 1
    return labeled


def merge_label(processed, data, label):
    """Extract data pixels with selected label"""
    # merge processed image with original
    stored_background_label = 1
    all_labels = np.unique(data)
    if len(all_labels) == 2:
        background_label = all_labels[all_labels != label][0]
        data[(processed == 0) & (data == label)] = background_label
    else:
        data[(processed == 0) & (original == curr_label)] = stored_background_label
    data[processed == 1] = curr_label
    return data
