# -*- coding: utf-8 -*-
# @Time : 2023/4/21 17:19
# @Author : Leqi Tian
# @File : setup.py

from setuptools import setup
from setuptools import find_packages


VERSION = '1.0.1'

setup(
    name='MetaMatching',  # package name
    version=VERSION,  # package version
    description='An integrated deep learning framework for the interpretation of untargeted metabolomics data',  
    packages=find_packages(),
    author='Leqi Tian',
    author_email='leqitian@link.cuhk.edu.cn',
    python_requires=">=3.9",
    zip_safe=False,
    install_requires=[
        'numpy>=1.21.5',
        'pandas>=1.5.3',
        'python_igraph>=0.10.4',
       'scikit_learn>=1.0.2',
        'torch>=1.11.0',
    ],
)