"""segmentify command line viewer.
"""

import argparse
import numpy  as np
import imageio
from napari import gui_qt
from segmentify import Viewer, util


def main(args):
    """The main Command Line Interface for Segmentify"""

    # parse in images
    imgs = [util.parse_img(img) for img in args.images]

    if len(imgs) > 1:
        imgs = np.stack(imgs, axis=0)
    else:
        imgs = np.array(imgs)

    with gui_qt():
        viewer = Viewer(imgs, heatmap=args.heatmap)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=str, help="Image to view and segment.")
    args = parser.parse_args()

    main(args)
