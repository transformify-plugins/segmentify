import os
from segmentify import Viewer, gui_qt, util

# parse input file
example_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "hpa.png")
img = util.parse_img(example_file)

with gui_qt():
    viewer = Viewer(img)

