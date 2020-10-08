

import numpy as np

from distutils.core import setup
from Cython.Build import cythonize


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


setup(
    ext_modules=cythonize("rbbox_overlap.pyx"),
    include_dirs=[numpy_include],
)
