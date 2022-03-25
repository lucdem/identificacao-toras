import os
import sys
from setuptools import setup

from Cython.Build import cythonize
import numpy

dir = sys.path[0]

setup(
	name='count_changes',
	ext_modules=cythonize(os.path.join(dir, "count_changes.pyx")),
	include_dirs=[numpy.get_include()],
	zip_safe=False
)

# call with: python count_changes/setup.py build_ext --build_lib count_changes