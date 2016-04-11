# python setup_cy.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy


""" How to make shared library from c code
http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html

export LD_LIBRARY_PATH='/home/jamessungjinkim/Dropbox/Aspuru-Guzik/python_lab/jamespy':$LD_LIBRARY_PATH

export CFLAGS="-I/home/jamessungjinkim/Dropbox/Aspuru-Guzik/python_lab/jamespy" 
export LDFLAGS="-L/home/jamessungjinkim/Dropbox/Aspuru-Guzik/python_lab/jamespy"

gcc -c -Wall -Werror -fpic jc.c
gcc -shared -o libjc.so jc.o

gcc -L/home/jamessungjinkim/Dropbox/Aspuru-Guzik/python_lab/jamespy-Wall test_jc.c -ljc
"""

ext_modules=[
	Extension("jpyx",
		sources=["*.pyx"],
		libraries=["jc"]
	)
]
 
setup(
	name = 'jpyx',
	ext_modules = cythonize( ext_modules)
)