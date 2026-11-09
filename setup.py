from setuptools import setup, find_packages

setup(
    name="s6-star-splitting",
    version="1.1.0",
    description="Generic Sixth-Order Commutator-Corrected Splitting Framework",
    author="Benjamin Veysi",
    author_email="matteolind2024@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "torch>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
