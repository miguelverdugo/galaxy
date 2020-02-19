# -*- coding: utf-8 -*-
"""
    Setup file for galaxy.
"""

import setuptools


with open('README.md') as f:
    __readme__ = f.read()

with open('LICENSE') as f:
    __license__ = f.read()


def setup_package():
    setuptools.setup(
        name='galaxy',
        version='0.0',
        description="minimalist galaxy model",
        long_description=__readme__,
        long_description_content_type='text/markdown',
        author='Miguel Verdugo',
        license="MIT",
        author_email='miguel.verdugo@univie.ac.at',
        url='https://github.com/miguelverdugo/galaxy',
        package_dir={'galaxy': 'galaxy'},
        packages=['galaxy'],
        package_data={'galaxy': ['galaxy/data/*']},
        include_package_data=True,
        install_requires=['numpy',
                          'astropy'],
# Also ScopeSim,  ScopeSim_Templates and speXtra
        classifiers=["Programming Language :: Python :: 3.7",
                        "License :: OSI Approved :: MIT License",
                        "Operating System :: OS Independent",
                        "Intended Audience :: Science/Research",
                        "Topic :: Scientific/Engineering :: Astronomy", ]
    )


if __name__ == "__main__":
    setup_package()


