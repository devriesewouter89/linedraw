#!/usr/bin/env python
from setuptools import setup

setup(
    name='linedraw',
    version='0.0.1',
    packages=["linedraw", 'linedraw.scripts'],
    scripts=['linedraw/linedraw.py'],
    license='MIT',
    install_requires=['opencv-contrib-python',
                      'numpy',
                      'Pillow'
                      ]
)
