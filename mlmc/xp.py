# This module provides xp,
# which is either cupy (if available)
# or else numpy.

try:
    import cupy as xp
except ImportError:
    # print('Tried to use cupy, but that\'s not available.')
    import numpy as xp