
import os
import numpy as np
import xarray as xr
from .options import OPTIONS
import pkg_resources


path2assets = pkg_resources.resource_filename("osnet", "assets/")


def add_MDT(ds: xr.Dataset, src=None) -> xr.Dataset:
    """ Add MDT variable to dataset

    This function is used by the facade to complement an input dataset with missing variables like MDT

    Parameters
    ----------
    ds: :class:`xarray.DataSet`
        Dataset with 'lat' and 'lon' coordinates to interpolate MDT on
    src: str
        Absolute path to the MDT netcdf source file

    Returns
    -------
    ds: :class:`xarray.DataSet`
        Dataset with new interpolated MDT variable
    """
    if src is None:
        path = os.path.join(path2assets, OPTIONS['mdt'])

    mdt = xr.open_dataset(path)
    mdt = mdt.where((mdt['longitude']>=OPTIONS['domain'][0])
                    & (mdt['longitude']<=OPTIONS['domain'][1])
                    & (mdt['latitude']>=OPTIONS['domain'][2])
                    & (mdt['latitude']<=OPTIONS['domain'][3]),
                    drop=True)
    mdt = mdt.interp(latitude=ds['lat'],
                     longitude=ds['lon'],
                     method = 'linear')['mdt'].astype(np.float32).squeeze().values.T
    if len(mdt.shape) == 0:
        mdt = mdt[np.newaxis, np.newaxis]
    if len(mdt.shape) == 1:
        mdt = mdt[np.newaxis]
    try:
        ds = ds.assign(variables={"MDT": (("lon", "lat"), mdt)})
    except Exception:
        ds = ds.assign(variables={"MDT": (("lon", "lat"), mdt.T)})
    finally:
        return ds
