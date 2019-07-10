"""
Perform interactive semantic segmentation
"""

"""
Display a labels layer above of an image layer using the add_labels and
add_image APIs
"""

from napari import Viewer, gui_qt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.morphology import watershed
from skimage.feature import peak_local_max

# Generate an initial image with blobs
blobs = data.binary_blobs(length=256, blob_size_fraction=0.1, n_dim=2,
                          volume_fraction=.2, seed=999)


# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(blobs)

local_maxi = peak_local_max(distance, indices=True, footprint=np.ones((5, 5)),
                            labels=blobs)

local_maxi_image = np.zeros(blobs.shape, dtype='bool')
for cord in local_maxi:
    local_maxi_image[tuple(cord)] = True
markers = ndi.label(local_maxi_image)[0]

labels = watershed(-distance, markers, mask=blobs)


with gui_qt():

    # create an empty viewer
    viewer = Viewer()

    # add the input image
    viewer.add_image(blobs.astype('float'), name='input', colormap='gray')

    # add the distance image
    viewer.add_image(distance, name='distance', colormap='gray')

    # add the resulting labels image
    viewer.add_labels(labels, name='output')

    # add the points
    viewer.add_points(local_maxi, face_color='blue', size=3, name='markers')

    @viewer.bind_key('r')
    def rerun(viewer):
        blobs = viewer.layers['input'].data
        distance = viewer.layers['distance'].data
        local_maxi = viewer.layers['markers'].data
        print('Number of markers: ', len(local_maxi))
        local_maxi_image = np.zeros(blobs.shape, dtype='bool')
        for cord in local_maxi:
            local_maxi_image[tuple(np.round(cord).astype(int))] = True
        markers = ndi.label(local_maxi_image)[0]
        labels = watershed(-distance, markers, mask=blobs)
            viewer.layers['output'].data = labels
