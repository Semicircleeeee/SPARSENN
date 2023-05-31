# -*- coding: utf-8 -*-
# @Time : 2023/5/21 17:19
# @Author : Leqi Tian
# @File : setup.py

from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as fh:
  long_description = fh.read()

VERSION = '1.6.0'

setup(
    name='meta_matching_tool',  # package name
    version=VERSION,  # package version
    description='An integrated deep learning framework for the interpretation of untargeted metabolomics data', 
    author='Leqi Tian',
    author_email='220049029@link.cuhk.edu.cn',
    url='https://github.com/tianlq-prog/SPARSENN',
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        'numpy>=1.21.5',
        'pandas>=1.5.3',
        'python_igraph>=0.10.4',
       'scikit_learn>=1.0.2',
        'torch >= 1.11.0',
    ],
    packages=find_packages(exclude=('Tutorial', 'docs', 'notebooks', 'real_data_preprocessing')),
    zip_safe=False,
)
