import sys
import os
import numpy as np
import xarray as xr
from .options import OPTIONS
import pkg_resources
import logging

log = logging.getLogger("osnet.utilities")

path2assets = pkg_resources.resource_filename("osnet", "assets/")

assets = { # Provide direct access to internal assets
    'mdt': xr.open_dataset(os.path.join(path2assets, OPTIONS['mdt'])),
    'bathy': xr.open_dataset(os.path.join(path2assets, OPTIONS['bathymetry'])),
}

def conv_lon(x):
    """ Make sure longitude axis is -180/180 """
    if np.all(np.logical_and(x>=0, x<=360)):
        x = np.where(x>180, x-360, x)
    return x


def add_MDT(ds: xr.Dataset, path=None) -> xr.Dataset:
    """ Add MDT variable to dataset

    This function is used by the facade to complement an input dataset with missing variables like MDT

    By default, the MDT_CNES_CLS18 Mean Dynamic Topography is interpolated on the lat/lon of the dataset.

    The horizontal resolution of the MDT is: 1/8 in latitude and longitude.

    The MDT_CNES_CLS18 Mean Dynamic Topography is calculated from the combination of altimetry, gravimetry
    (including GOCE and GRACE) and in-situ data. The reference time-period is 1993-2012.

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

    # Preserve attributes for the output:
    attrs = mdt['mdt'].attrs
    for k in ['institution', 'processing_level', 'product_version', 'summary', 'title']:
        attrs[k] = mdt.attrs[k]

    # Squeeze domain
    mdt = mdt.where((mdt['longitude']>=conv_lon(OPTIONS['domain'][0]))
                    & (mdt['longitude']<=conv_lon(OPTIONS['domain'][1]))
                    & (mdt['latitude']>=OPTIONS['domain'][2])
                    & (mdt['latitude']<=OPTIONS['domain'][3]),
                    drop=True)
    # Interp on the input grid:
    mdt = mdt.interp(latitude=ds['lat'],
                     longitude=conv_lon(ds['lon']),
                     method = 'linear')['mdt'].astype(np.float32).squeeze().values.T
    if len(mdt.shape) == 0:
        mdt = mdt[np.newaxis, np.newaxis]
    if len(mdt.shape) == 1:
        mdt = mdt[np.newaxis]
    try:
        ds = ds.assign(variables={"MDT": (("lon", "lat"), mdt)})
        ds['MDT'].attrs = attrs
    except Exception:
        ds = ds.assign(variables={"MDT": (("lon", "lat"), mdt.T)})
        ds['MDT'].attrs = attrs
    finally:
        return ds


def add_BATHY(ds: xr.Dataset, path=None) -> xr.Dataset:
    """ Add BATHY variable to dataset

    This function is used by the facade to complement an input dataset with missing variables like BATHY

    By default, the ETOPO1_Ice_g_gmt4 bathymetry is interpolated on the lat/lon of the dataset.

    The horizontal resolution of the bathymetry is: 1/60 in latitude and longitude.

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

    # Preserve attributes for the output:
    attrs = bathy['bathymetry'].attrs
    for k in ['title', 'GMT_version']:
        attrs[k] = bathy.attrs[k]
    attrs['standard_name'] = 'sea_floor_depth_below_sea_surface'

    # Squeeze domain:
    bathy = bathy.where((bathy['longitude']>=conv_lon(OPTIONS['domain'][0]))
                    & (bathy['longitude']<=conv_lon(OPTIONS['domain'][1]))
                    & (bathy['latitude']>=OPTIONS['domain'][2])
                    & (bathy['latitude']<=OPTIONS['domain'][3]),
                    drop=True)
    # Interp
    bathy = bathy.interp(latitude=ds['lat'],
                         longitude=conv_lon(ds['lon']),
                         method = 'linear')['bathymetry'].astype(np.float32).squeeze().values.T
    if len(bathy.shape) == 0:
        bathy = bathy[np.newaxis, np.newaxis]
    if len(bathy.shape) == 1:
        bathy = bathy[np.newaxis]
    try:
        ds = ds.assign(variables={"BATHY": (("lon", "lat"), bathy)})
        ds['BATHY'].attrs = attrs
    except Exception:
        ds = ds.assign(variables={"BATHY": (("lon", "lat"), bathy.T)})
        ds['BATHY'].attrs = attrs
    finally:
        return ds


def add_SSTclim(ds: xr.Dataset, path=None) -> xr.Dataset:
    """ Add OSTIA SST climatology to dataset

    This function is used by the facade to complement an input dataset with missing variables like SST

    This SST is from the Global Ocean OSTIA Sea Surface Temperature dataset:
    https://resources.marine.copernicus.eu/product-detail/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011/INFORMATION

    The horizontal resolution of the climatological SST is: 1/20 in latitude and longitude.

    The SST climatology is calculated from daily fields over 1993-2019.

    Parameters
    ----------
    ds: :class:`xarray.DataSet`
        Dataset with 'lat' and 'lon' coordinates to interpolate SST on
    path: str
        Absolute path to the SST netcdf source file

    Returns
    -------
    ds: :class:`xarray.DataSet`
        Dataset with new interpolated SST climatology variable
    """
    if path is None:
        path = os.path.join(path2assets, OPTIONS['sst_clim'])
    ds_src = xr.open_dataset(path)

    # Preserve attributes for the output:
    attrs = ds_src['analysed_sst'].attrs
    for k in ['Conventions', 'title', 'summary', 'institution', 'references', 'product_version']:
        attrs[k] = ds_src.attrs[k]

    # Squeeze domain
    ds_src = ds_src.where((ds_src['lon']>=conv_lon(OPTIONS['domain'][0]))
                    & (ds_src['lon']<=conv_lon(OPTIONS['domain'][1]))
                    & (ds_src['lat']>=OPTIONS['domain'][2])
                    & (ds_src['lat']<=OPTIONS['domain'][3]),
                    drop=True)

    # Interp on the input grid:
    field = ds_src.interp(lat=ds['lat'],
                     lon=conv_lon(ds['lon']),
                     method = 'linear')['analysed_sst'].astype(np.float32).squeeze().values.T

    field = field-273.15
    attrs['units'] = 'degC'
    attrs['valid_min'] = -2
    attrs['valid_max'] = 40

    # Reshape
    if len(field.shape) == 0:
        field = field[np.newaxis, np.newaxis]
    if len(field.shape) == 1:
        field = field[np.newaxis]
    try:
        ds = ds.assign(variables={"SST": (("lon", "lat"), field)})
        ds['SST'].attrs = attrs
    except Exception:
        ds = ds.assign(variables={"SST": (("lon", "lat"), field.T)})
        ds['SST'].attrs = attrs
    finally:
        return ds


def add_SLAclim(ds: xr.Dataset, path=None) -> xr.Dataset:
    """ Add AVISO SLA climatology, and related fields, to dataset

    This function is used by the facade to complement an input dataset with missing variables like SST

    This SLA is from the global SSALTO/DUACS Sea Surface Height measured by Altimetry dataset.
    https://resources.marine.copernicus.eu/product-detail/SEALEVEL_GLO_PHY_MDT_008_063/INFORMATION

    The horizontal resolution of the climatological SLA is: 1/4 in latitude and longitude.

    The SLA climatology is calculated from daily fields over 1993-2019.

    Parameters
    ----------
    ds: :class:`xarray.DataSet`
        Dataset with 'lat' and 'lon' coordinates to interpolate SLA on
    path: str
        Absolute path to the SST netcdf source file

    Returns
    -------
    ds: :class:`xarray.DataSet`
        Dataset with new interpolated SST climatology variable
    """
    if path is None:
        path = os.path.join(path2assets, OPTIONS['sla_clim'])
    ds_src = xr.open_dataset(path)

    # Preserve attributes for the output:
    attrs = ds_src['sla'].attrs
    for k in ['Conventions', 'title', 'summary', 'institution', 'references', 'product_version']:
        attrs[k] = ds_src.attrs[k]

    # Squeeze domain
    ds_src = ds_src.where((ds_src['longitude']>=conv_lon(OPTIONS['domain'][0]))
                    & (ds_src['longitude']<=conv_lon(OPTIONS['domain'][1]))
                    & (ds_src['latitude']>=OPTIONS['domain'][2])
                    & (ds_src['latitude']<=OPTIONS['domain'][3]),
                    drop=True)

    # Interp all variables:
    def interp_this(ds, x, vname='sla'):
        x = x.interp(latitude=ds['lat'],
                     longitude=conv_lon(ds['lon']),
                     method = 'linear')[vname].astype(np.float32).squeeze().values.T
        return x

    for v in ['sla', 'ugosa', 'vgosa', 'ugos', 'vgos']:
        val = interp_this(ds, ds_src, vname=v)
        if len(val.shape) == 0:
            val = val[np.newaxis, np.newaxis]
        if len(val.shape) == 1:
            val = val[np.newaxis]
        try:
            ds = ds.assign(variables={v.upper(): (("lon", "lat"), val)})
        except Exception:
            ds = ds.assign(variables={v.upper(): (("lon", "lat"), val.T)})
        ds[v.upper()].attrs = attrs

    # Output
    return ds


def check_and_complement(ds: xr.Dataset) -> xr.Dataset:
    """ Check input dataset and possibly add missing variables

    This function is used by the facade when asked to make a prediction

    Here we check if the input :class:`xarray.DataSet` has the required variables to make a prediction:
        1. ``lat``
        1. ``lon``
        1. ``time`` or ``dayOfYear``
        1. ``BATHY``
        1. ``MDT``
        1. ``SST``
        1. ``SLA``
        1. ``UGOSA``
        1. ``VGOSA``
        1. ``UGOS``
        1. ``VGOS``

    If ``BATHY``, ``MDT`` or ``dayOfYear`` are missing, we try to add them internally.
    If ``SST`` is missing, we use a 1993-2019 climatology.
    If ``SLA`` and related variables are missing, we use a 1993-2019 climatology.

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

    if "SST" not in ds.data_vars:
        try:
            ds = add_SSTclim(ds)
            log.debug("Added new variable '%s' to dataset" % "SSTclim")
            added.append('SST')
        except Exception:
            raise ValueError("SST is missing from input and cannot be added automatically")

    if "SLA" not in ds.data_vars:
        try:
            ds = add_SLAclim(ds)
            log.debug("Added new variable '%s' to dataset" % "SLAclim")
            [added.append(v) for v in ['SLA', 'UGOSA', 'VGOSA', 'UGOS', 'VGOS']]
        except Exception:
            raise ValueError("SLA is missing from input and cannot be added automatically")

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
        # This will be used by the facade to possible delete these variables from the output dataset

    return ds


def get_sys_info():
    "Returns system information as a dict"

    blob = []

    # get full commit hash
    commit = None
    if os.path.isdir(".git") and os.path.isdir("osnet"):
        try:
            pipe = subprocess.Popen(
                'git log --format="%H" -n 1'.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            so, serr = pipe.communicate()
        except Exception:
            pass
        else:
            if pipe.returncode == 0:
                commit = so
                try:
                    commit = so.decode("utf-8")
                except ValueError:
                    pass
                commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    try:
        (sysname, nodename, release, version_, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", sys.version),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "%s" % (sysname)),
                ("OS-release", "%s" % (release)),
                ("machine", "%s" % (machine)),
                ("processor", "%s" % (processor)),
                ("byteorder", "%s" % sys.byteorder),
                ("LC_ALL", "%s" % os.environ.get("LC_ALL", "None")),
                ("LANG", "%s" % os.environ.get("LANG", "None")),
                ("LOCALE", "%s.%s" % locale.getlocale()),
            ]
        )
    except Exception:
        pass

    return blob


def netcdf_and_hdf5_versions():
    libhdf5_version = None
    libnetcdf_version = None
    try:
        import netCDF4

        libhdf5_version = netCDF4.__hdf5libversion__
        libnetcdf_version = netCDF4.__netcdf4libversion__
    except ImportError:
        try:
            import h5py

            libhdf5_version = h5py.version.hdf5_version
        except ImportError:
            pass
    return [("libhdf5", libhdf5_version), ("libnetcdf", libnetcdf_version)]


def show_versions(file=sys.stdout):  # noqa: C901
    """ Print the versions of osnet and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    sys_info = get_sys_info()

    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except Exception as e:
        print(f"Error collecting netcdf / hdf5 version: {e}")

    deps = [
        # (MODULE_NAME, f(mod) -> mod version)
        ("osnet", lambda mod: mod.__version__),
        ("tensorflow", lambda mod: mod.__version__),
        ("keras", lambda mod: mod.__version__),
        ("joblib", lambda mod: mod.__version__),
        ("numba", lambda mod: mod.__version__),
        ("dask", lambda mod: mod.__version__),

        ("numpy", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("xarray", lambda mod: mod.__version__),
        ("sklearn", lambda mod: mod.__version__),

        ("matplotlib", lambda mod: mod.__version__),
        ("gsw", lambda mod: mod.__version__),
        ("seaborn", lambda mod: mod.__version__),
        ("IPython", lambda mod: mod.__version__),
        ("netCDF4", lambda mod: mod.__version__),

        ("packaging", lambda mod: mod.__version__),
        ("pip", lambda mod: mod.__version__),
    ]

    deps_blob = list()
    for (modname, ver_f) in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except Exception:
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
                deps_blob.append((modname, ver))
            except Exception:
                deps_blob.append((modname, "installed"))

    print("\nINSTALLED VERSIONS", file=file)
    print("------------------", file=file)

    for k, stat in sys_info:
        print(f"{k}: {stat}", file=file)

    print("", file=file)
    for k, stat in deps_blob:
        print(f"{k}: {stat}", file=file)
