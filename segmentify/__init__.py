__version__ = '0.1.0'

from .gui import segmentation
from . import _key_bindings

# Note that importing _key_bindings is needed as the Labels layer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and so is deleted below
del _key_bindings
