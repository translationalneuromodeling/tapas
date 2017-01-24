#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Setup for sooner module.

'''

#from distutils.core import setup
import os
import glob
import platform
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
import numpy as np

if platform.system().lower() == 'darwin':
    os.environ['CC'] = 'clang-omp'

openmp = {'extra_compile_args': ['-fopenmp'],
    'extra_link_args' : ['-fopenmp']}

llh = Extension('tapas.sem.wrapped',
        ['./tapas/sem/antisaccades/wrapped.pyx'] + glob.glob('./src/antisaccades/*.c'),
        libraries=['gsl', 'gslcblas'],
        library_dirs=['/usr/local/lib', '/opt/local/libs'], # For mac
        include_dirs=[np.get_include(), '/opt/local/include', '.'],
        define_macros=[('TAPAS_PYTHON', None)],
        **openmp)


if __name__ == '__main__':
    setup(name='tapas.sem',
        zip_safe=False,
        version='1.0',
        requires=['cython', 'numpy'],
        description='Python packages for Saccadic Eye movement Models',
        author='Eduardo Aponte',
        author_email='aponteeduardo@gmail.com',
        url='www.translationalneuromodeling.org/tapas/',
        packages=find_packages(),
        license='GPLV.3',
        cmdclass={'build_ext': build_ext},
        ext_modules=[llh],
        namespace_package=['tapas']
        )

if __name__ == '__main__':
    pass

