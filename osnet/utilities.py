import sys
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

    Here we check of the input :class:`xarray.DataSet` has the required variables to make a prediction:
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
