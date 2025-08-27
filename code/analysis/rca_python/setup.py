#!/usr/bin/env python
"""
Setup script for rca_python package.
"""

from setuptools import setup, find_packages

setup(
    name="rca_python",
    version="1.0.0",
    description="A Python implementation of Reliable Components Analysis (RCA) for neural data analysis",
    long_description="RCA Python package designed for integration with music preference EEG study. Based on the MATLAB toolbox by Jacek P. Dmochowski (2015).",
    author="Music Preference Study Team", 
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "mne>=1.0.0",
        "h5py",
        "pandas>=1.3.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)