# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:09:30 2024

@author: zhoushus
"""

from setuptools import setup, find_packages

setup(
    name="lightgbm_feature_importance_evaluator_zhoumath",
    version="0.1.1",
    description="Evaluate feature importance for LightGBM models using various methods.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Zhoushus",
    author_email="zhoushus@foxmail.com",
    url="https://github.com/shanghaizhoushus/lightgbm_feature_importance_evaluator_zhoumath",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "shap",
        "lightgbm"
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)