"""
Setup script for datasketch package
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

test_requires = ['coverage', 'mock>=2.0.0']

setup(
    name='datasketch',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html

    version='1.2.7',
  
    description='Probabilistic data structures for processing and searching very large datasets',
    long_description=long_description,

    # The project's main homepage.
    url='https://ekzhu.github.io/datasketch',

    # Author details
    author='ekzhu',
    author_email='ekzhu@cs.toronto.edu',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Database',
        'Topic :: Scientific/Engineering :: Information Analysis',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='database datamining',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'test*', 'benchmarks', 'examples', 'docsrc']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.11', 'redis>=2.10.0'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test,aio,aio-test]
    extras_require={
        'dev': ['check-manifest'],
        'test': test_requires + ['mockredispy', "aiounittest ; python_version>='3.6'", "aioredis ; python_version>='3.6'", "motor ; python_version>='3.6'", "pymongo ; python_version>='3.6'"],
        'experimental_aio': ["aiounittest ; python_version>='3.6'", "aioredis ; python_version>='3.6'", "motor ; python_version>='3.6'", "pymongo ; python_version>='3.6'"],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
    #    'sample': ['package_data.dat'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    },
)
