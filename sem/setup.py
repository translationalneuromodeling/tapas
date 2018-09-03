#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Setup for sooner module.

'''

import glob
import platform
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
import numpy as np

if platform.system().lower() == 'darwin':
    #os.environ['CC'] = 'clang'
    openmp = {}
    extra_libs = []
else:
    openmp = {
        'extra_compile_args': ['-fopenmp'],
        'extra_link_args' : ['-fopenmp']}
    extra_libs = ['gomp']


llh = Extension(
        'tapas.sem.antisaccades.wrappers',
        ['./tapas/sem/antisaccades/wrappers.pyx'] + \
                glob.glob('./src/antisaccades/*.c'),
        libraries=['gsl', 'gslcblas'] + extra_libs,
        library_dirs=['/usr/local/lib', '/opt/local/libs'],  # For mac
        include_dirs=[np.get_include(), '/opt/local/include', '.', './src/',
            '/usr/local/opt/llvm/include/clang/'],
        define_macros=[('TAPAS_PYTHON', None)],
        **openmp)


if __name__ == '__main__':
    setup(
        name='tapas.sem',
        zip_safe=False,
        version='1.1',
        requires=['cython', 'numpy'],
        install_requirement=['gsl>=1.6.0'],
        description='Python packages for Saccadic Eye movement Models',
        author='Eduardo Aponte',
        author_email='aponteeduardo@gmail.com',
        url='www.translationalneuromodeling.org/tapas/',
        packages=find_packages(),
        license='GPLV.3',
        cmdclass={'build_ext': build_ext},
        ext_modules=[llh],
        namespace_packages=['tapas']
        )
