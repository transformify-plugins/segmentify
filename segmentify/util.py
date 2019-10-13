import imageio
import numpy as np
import numba
import math
from scipy import stats

def parse_img(img):
    """parse in a input image into the standardized format for segmentify

    Parameters
    ----------
    img : np.array
        The image to be parsed

    Returns
    -------
    The parsed image
    """

    # read in image
    img = imageio.imread(img)

    # check dimention order
    if img.shape[-1] <=5:
        img = np.transpose(img, (2,0,1))

    # normalize to 0 - 1
    if img.max() > 1:
        img = img / 255.

    # use only one dimention
    if img.shape[0] > 1:
        img = np.mean(img, axis=0)

    return img

@numba.jit(nopython=True, fastmath=True, cache=True)
def _get_mode(target_region):
    """get the mode of list

    Parameters:
    -----------
    target_region: list(int)
        A list of pixels

    Returns
    -------
    The mode of a given list
    """
    counter = 0
    target = target_region[0]
    for t in target_region:
        curr_freq = target_region.count(t)
        if curr_freq > counter:
            counter = curr_freq
            target = t
    return target

@numba.jit(fastmath=True, cache=True)
def _erode_img(img, target_label):
    """multi-class image erosion

    This function performs a multi class by finding the mode in a sliding kernel

    Parameters
    ----------
    img: np.array
        The image to be eroded
    target_label: int
        The label of the pixels to be eroded

    Returns
    -------
    The eroded image
    """

    rows, cols = img.shape

    # padd image with sides of original image
    padded_img = np.zeros((rows+2, cols+2))
    padded_img[1:-1,1:-1] = img
    padded_img[:,0] = padded_img[:,1]
    padded_img[:,-1] = padded_img[:,-2]
    padded_img[0,:] = padded_img[1,:]
    padded_img[-1,:] = padded_img[-2,:]

    output_img = np.zeros_like(img)

    for r in range(rows):
        for c in range(cols):
            # use a 3x3 square sliding window
            region = np.copy(padded_img[r:r+3, c:c+3])
            region_flattern = np.copy(np.reshape(region,-1))

            # use the target label if no other labels exisit in the sliding window
            if np.all(region_flattern == target_label):
                output_img[r,c] = target_label
            else:
                target_region = [v for v in region_flattern \
                                 if v != target_label]

                if len(target_region) == 0:
                    continue

                target = _get_mode(target_region)
                output_img[r,c] = target
    return output_img


def _norm_entropy(probs):
    """get the normalized entropy based on a list of proabilities

    Parameters
    ----------
    probs: list
        list of probabilities

    Returns
    -------
    normalized entropy of the probabilities
    """

    entropy = 0
    for prob in probs:
        if prob > 0:
            entropy += prob * math.log(prob, math.e)
        else:
            entropy += 0
    return - entropy / len(probs)
