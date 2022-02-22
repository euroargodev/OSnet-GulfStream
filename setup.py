# -*coding: UTF-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='osnet',
    version='0.1.0',
    author="osnet Developers",
    author_email="etienne.pauthenet@ifremer.fr",
    description="A python library to make predictions with OSnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/euroargodev/OSnet-GulfStream",
    packages=setuptools.find_packages(),
    package_dir={'osnet': 'osnet'},
    package_data={'osnet': ['assets/*.nc']},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ]
)
