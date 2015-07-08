# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from distutils.extension import Extension


ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():
    sources = [os.path.join(ROOT, 'cextinction.pyx')]
    modeling_ext = Extension(
        name="astropy.modeling.cextinction",
        include_dirs=["numpy"],
        sources=sources)
    return [modeling_ext]


def get_package_data():
    return {
        'astropy.modeling.tests': ['data/*.fits', 'data/*.hdr',
                                   '../../wcs/tests/maps/*.hdr']
    }


def requires_2to3():
    return False
