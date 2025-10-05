from setuptools import setup, find_packages

setup(
    name="endpoint-cfm",
    version="0.1.0",
    description="Conditional Flow Matching for Endpoint Prediction in Dynamical Systems",
    author="",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.10.0",
        "pytorch-lightning>=2.0.0",
        "hydra-core>=1.2.0",
        "omegaconf>=2.2.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0",
        "torchcfm>=1.0.0",
        "scikit-learn>=1.0.0",
        "pathlib",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "endpoint-cfm-train=endpoint_cfm.cli:train_cli",
            "endpoint-cfm-infer=endpoint_cfm.cli:infer_cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
