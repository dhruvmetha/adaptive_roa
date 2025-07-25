from setuptools import setup, find_packages

setup(
    name="olympics_classifier",
    version="0.1.0",
    description="ai olympics classifier",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "hydra-core",
        "numpy",
        "tqdm",
        "torchvision",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "train-classifier=train_classifier:main",
        ],
    },
)
