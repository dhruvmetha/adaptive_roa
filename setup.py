from setuptools import setup, find_packages

setup(
    name="adaptive_roa",
    packages=find_packages(),
    install_requires=[
        "gpytorch>=1.11",  # partx: variational GP classifier surrogate
    ],
)
