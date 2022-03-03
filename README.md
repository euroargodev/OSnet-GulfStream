|<img src="https://github.com/euroargodev/OSnet-GulfStream/raw/main/docs/_static/osnet_landscape.png" width="300px"/><h1>Gulf-Stream</h1>|``OSnet`` is a python library to make T/S/MLD predictions in the Gulf Stream region using Neural Network|
|:---------:|:-------|
|Methodology| Pauthenet et al, 2022 (insert link to paper here)<br>Method source code: https://github.com/euroargodev/OSnet|
|OSnet gridded dataset|[![Gridded dataset](https://zenodo.org/badge/DOI/10.5281/zenodo.6011144.svg)](https://doi.org/10.5281/zenodo.6011144)
|OSnet software<br><i>to make your own predictions</i>|![License](https://img.shields.io/github/license/euroargodev/argopy) [![Python version](https://img.shields.io/pypi/pyversions/argopy)](//pypi.org/project/argopy/)<br>[![](https://img.shields.io/github/release-date/euroargodev/osnet)](//github.com/euroargodev/osnet/releases) [![PyPI](https://img.shields.io/pypi/v/osnet)](//pypi.org/project/osnet/) |

# Documentation

The folder ``docs`` contains many advanced notebooks to get you started with making your OSnet predictions.

# Install

```bash
pip install git+http://github.com/euroargodev/OSnet-GulfStream.git
```

To create the ``OSnet`` python development environment, you can:

```bash
conda env create -f ci/requirements/py3.8-dev.yml
```

Then, to make it available in Jupyter notebooks:

```bash
python -m ipykernel install --user --name=OSnet
```

# Usage

[![Binder](https://img.shields.io/static/v1.svg?logo=Jupyter&label=Binder&message=Click+here+to+try+osnet+online+!&color=blue)](https://mybinder.org/v2/gh/euroargodev/binder-sandbox/pangeo-ml?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Feuroargodev%252FOSnet-GulfStream%26urlpath%3Dlab%252Ftree%252FOSnet-GulfStream%252Fdocs%252Fdemo-predictions.ipynb%26branch%3Dmain)

Import library:
```python
import osnet
import xarray as xr
```

Prepare inputs:
```python
ds_in = xr.DataSet([...])
```

Load model and make prediction:
```python
model = osnet.load('Gulf-Stream')
ds_out = model.predict(ds_in)
ds_out = model.predict(ds_in, adjust_mld=False)  # Do not perform MLD adjustment
```

***
This repository has been developed at the Laboratory for Ocean Physics and Satellite remote sensing, Ifremer, within the framework of the Euro-ArgoRISE project. This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement no 824131. Call INFRADEV-03-2018-2019: Individual support to ESFRI and other world-class research infrastructures.

<p align="center">
<a href="https://www.euro-argo.eu/EU-Projects/Euro-Argo-RISE-2019-2022">
<img src="https://user-images.githubusercontent.com/59824937/146353317-56b3e70e-aed9-40e0-9212-3393d2e0ddd9.png" height="75"/>
</a>
<a href="https://www.umr-lops.fr">
<img src="https://user-images.githubusercontent.com/59824937/146353157-b45e9943-9643-45d0-bab5-80c22fc2d889.jpg" height="75"/>
</a>
<a href="https://wwz.ifremer.fr">
<img src="https://user-images.githubusercontent.com/59824937/146353099-bcd2bd4e-d310-4807-aee2-9cf24075f0c3.jpg" height="75"/>
</a>
</p>