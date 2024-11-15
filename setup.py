from setuptools import setup, find_packages

setup(
    name="lightgbm_feature_importance_evaluator_zhoumath",  # Package name
    version="0.1.0",                                       # Package version
    description="Evaluate feature importance for LightGBM models using various methods.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Zhoushus",
    author_email="zhoushus@foxmail.com",
    url="https://github.com/shanghaizhoushus/lightgbm_feature_importance_evaluator_zhoumath",  # GitHub URL
    packages=find_packages(),                            # Automatically find subpackages
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "shap",
        "lightgbm",
    ],
    python_requires=">=3.7",                             # Minimum supported Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)