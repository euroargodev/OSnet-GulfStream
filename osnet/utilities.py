
import os
import numpy as np
import xarray as xr
from .options import OPTIONS
import pkg_resources
import logging

log = logging.getLogger("osnet.utilities")

path2assets = pkg_resources.resource_filename("osnet", "assets/")


def add_MDT(ds: xr.Dataset, path=None) -> xr.Dataset:
    """ Add MDT variable to dataset

    This function is used by the facade to complement an input dataset with missing variables like MDT

    Parameters
    ----------
    ds: :class:`xarray.DataSet`
        Dataset with 'lat' and 'lon' coordinates to interpolate MDT on
    path: str
        Absolute path to the MDT netcdf source file

    Returns
    -------
    ds: :class:`xarray.DataSet`
        Dataset with new interpolated MDT variable
    """
    if path is None:
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


def add_BATHY(ds: xr.Dataset, path=None) -> xr.Dataset:
    """ Add BATHY variable to dataset

    This function is used by the facade to complement an input dataset with missing variables like BATHY

    Parameters
    ----------
    ds: :class:`xarray.DataSet`
        Dataset with 'lat' and 'lon' coordinates to interpolate BATHY on
    path: str
        Absolute path to the BATHY netcdf source file

    Returns
    -------
    ds: :class:`xarray.DataSet`
        Dataset with new interpolated BATHY variable
    """
    if path is None:
        path = os.path.join(path2assets, OPTIONS['bathymetry'])

    bathy = xr.open_dataset(path)
    bathy = bathy.where((bathy['longitude']>=OPTIONS['domain'][0])
                    & (bathy['longitude']<=OPTIONS['domain'][1])
                    & (bathy['latitude']>=OPTIONS['domain'][2])
                    & (bathy['latitude']<=OPTIONS['domain'][3]),
                    drop=True)
    bathy = bathy.interp(latitude=ds['lat'],
                         longitude=ds['lon'],
                         method = 'linear')['bathymetry'].astype(np.float32).squeeze().values.T
    if len(bathy.shape) == 0:
        bathy = bathy[np.newaxis, np.newaxis]
    if len(bathy.shape) == 1:
        bathy = bathy[np.newaxis]
    try:
        ds = ds.assign(variables={"BATHY": (("lon", "lat"), bathy)})
    except Exception:
        ds = ds.assign(variables={"BATHY": (("lon", "lat"), bathy.T)})
    finally:
        return ds


def check_and_complement(ds: xr.Dataset) -> xr.Dataset:
    """ Check input dataset and possibly add missing variables

    This function is used by the facade when asked to make a prediction

    Parameters
    ----------
    ds: :class:`xarray.DataSet`
        Dataset with at least 'lat', 'lon'  and 'time' coordinates

    Returns
    -------
    ds: :class:`xarray.DataSet`
        Dataset suitable for OSnet predictions

    """
    added = []
    if "BATHY" not in ds.data_vars:
        try:
            ds = add_BATHY(ds)
            log.debug("Added new variable '%s' to dataset" % "BATHY")
            added.append('BATHY')
        except Exception:
            raise ValueError("BATHY is missing from input and cannot be added automatically")

    if "MDT" not in ds.data_vars:
        try:
            ds = add_MDT(ds)
            log.debug("Added new variable '%s' to dataset" % "MDT")
            added.append('MDT')
        except Exception:
            raise ValueError("MDT is missing from input and cannot be added automatically")

    if "dayOfYear" not in ds.data_vars:
        if "time" not in ds.dims:
            raise ValueError("Input dataset must have a 'time' coordinate/dimension")
        try:
            ds = ds.assign(variables={"dayOfYear": ds['time.dayofyear']})
            log.debug("Added new variable '%s' to dataset" % "dayOfYear")
            added.append('dayOfYear')
        except Exception:
            raise ValueError("dayOfYear is missing from input and cannot be added automatically")

    if len(added) > 0:
        ds.attrs['OSnet-added'] = ";".join(added)

    return ds