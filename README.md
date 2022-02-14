## OSnet: Gulf Stream

This repository contains a python library to make T/S profiles prediction using the OSnet model computed for the GUlf Stream region in Pauthenet et al (in prep.).

If you're looking for the code to develop and train the OSnet model, please visit: https://github.com/euroargodev/OSnet

# Install

```bash
pip instal osnet
```

To create the ``OSnet`` python environment suitable to run these notebooks, you should:

```bash
conda env create -f environment.yml
```

Then, to make it available in Jupyter notebooks:

```bash
python -m ipykernel install --user --name=OSnet
```

# Usage

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
model = osnet('Gulf-Stream')
ds_out = model.predict(ds_in)
ds_out = model.predict(ds_in, adjust_mld=False)  # Do not perform MLD adjustment
ds_out = model.predict(ds_in, id=4)  # Only use model ID 4
ds_out = model.predict(ds_in, id=[1,2])  # Only use model ID 1 and 2
```