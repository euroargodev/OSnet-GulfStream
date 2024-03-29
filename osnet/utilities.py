import sys
import os
import numpy as np
import xarray as xr
from .options import OPTIONS
import pkg_resources
import logging
import importlib
import locale
import platform
import struct
import subprocess
import pandas as pd

log = logging.getLogger("osnet.utilities")

path2assets = pkg_resources.resource_filename("osnet", "assets/")

assets = {  # Provide simple user access to internal assets
    "mdt": xr.open_dataset(os.path.join(path2assets, OPTIONS["mdt"])),
    "bathy": xr.open_dataset(os.path.join(path2assets, OPTIONS["bathymetry"])),
    "sst_clim": xr.open_dataset(os.path.join(path2assets, OPTIONS["sst_clim"])),
    "sla_clim": xr.open_dataset(os.path.join(path2assets, OPTIONS["sla_clim"])),
}


def conv_lon(x):
    """ Make sure longitude axis is -180/180 """
    if np.all(np.logical_and(x >= 0, x <= 360)):
        x = np.where(x > 180, x - 360, x)
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
        path = os.path.join(path2assets, OPTIONS["mdt"])
    mdt = xr.open_dataset(path)

    # Preserve attributes for the output:
    attrs = mdt["mdt"].attrs
    for k in ["institution", "processing_level", "product_version", "summary", "title"]:
        attrs[k] = mdt.attrs[k]

    # Squeeze domain
    if not OPTIONS["unbound"]:
        mdt = mdt.where(
            (mdt["longitude"] >= conv_lon(OPTIONS["domain"][0]))
            & (mdt["longitude"] <= conv_lon(OPTIONS["domain"][1]))
            & (mdt["latitude"] >= OPTIONS["domain"][2])
            & (mdt["latitude"] <= OPTIONS["domain"][3]),
            drop=True,
        )
    # Interp on the input grid:
    if ds["lat"].dims == (
        "sampling",
    ):  # todo this is clearly not safe proof to any kind of inputs and need precise doc
        mdt = (
            mdt.interp(
                latitude=ds["lat"], longitude=conv_lon(ds["lon"]), method="linear"
            )["mdt"]
            .astype(np.float32)
            .values
        )
        ds = ds.assign(variables={"MDT": (("sampling"), mdt)})
    else:
        mdt = (
            mdt.interp(
                latitude=ds["lat"], longitude=conv_lon(ds["lon"]), method="linear"
            )["mdt"]
            .astype(np.float32)
            .squeeze()
            .values.T
        )
        if len(mdt.shape) == 0:
            mdt = mdt[np.newaxis, np.newaxis]
        if len(mdt.shape) == 1:
            mdt = mdt[np.newaxis]
        try:
            ds = ds.assign(variables={"MDT": (("lon", "lat"), mdt)})
        except Exception:
            ds = ds.assign(variables={"MDT": (("lon", "lat"), mdt.T)})

    #
    ds["MDT"].attrs = attrs
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
        path = os.path.join(path2assets, OPTIONS["bathymetry"])
    bathy = xr.open_dataset(path)

    # Preserve attributes for the output:
    attrs = bathy["bathymetry"].attrs
    for k in ["title", "GMT_version"]:
        attrs[k] = bathy.attrs[k]
    attrs["standard_name"] = "sea_floor_depth_below_sea_surface"

    # Squeeze domain:
    if not OPTIONS["unbound"]:
        bathy = bathy.where(
            (bathy["longitude"] >= conv_lon(OPTIONS["domain"][0]))
            & (bathy["longitude"] <= conv_lon(OPTIONS["domain"][1]))
            & (bathy["latitude"] >= OPTIONS["domain"][2])
            & (bathy["latitude"] <= OPTIONS["domain"][3]),
            drop=True,
        )
    # Interp
    if ds["lat"].dims == (
        "sampling",
    ):  # todo this is clearly not safe proof to any kind of inputs and need precise doc
        # log.debug("Input with 'sampling' dimension")
        # log.debug(bathy)
        bathy = (
            bathy.interp(
                latitude=ds["lat"], longitude=conv_lon(ds["lon"]), method="linear"
            )["bathymetry"]
            .astype(np.float32)
            .values
        )
        ds = ds.assign(variables={"BATHY": (("sampling"), bathy)})
    else:
        # log.debug("Input is with 'lat/lon' dimensions")
        # log.debug(bathy)
        bathy = (
            bathy.interp(
                latitude=ds["lat"], longitude=conv_lon(ds["lon"]), method="linear"
            )["bathymetry"]
            .astype(np.float32)
            .squeeze()
            .values.T
        )
        if len(bathy.shape) == 0:
            bathy = bathy[np.newaxis, np.newaxis]
        if len(bathy.shape) == 1:
            bathy = bathy[np.newaxis]
        try:
            ds = ds.assign(variables={"BATHY": (("lon", "lat"), bathy)})
        except Exception:
            ds = ds.assign(variables={"BATHY": (("lon", "lat"), bathy.T)})
    #
    # log.debug(bathy)
    ds["BATHY"].attrs = attrs
    ds['BATHY'].attrs['units'] = 'm'
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
        path = os.path.join(path2assets, OPTIONS["sst_clim"])
    ds_src = xr.open_dataset(path)

    # Preserve attributes for the output:
    attrs = ds_src["analysed_sst"].attrs
    for k in [
        "Conventions",
        "title",
        "summary",
        "institution",
        "references",
        "product_version",
    ]:
        attrs[k] = ds_src.attrs[k]

    # Squeeze domain
    ds_src = ds_src.where(
        (ds_src["lon"] >= conv_lon(OPTIONS["domain"][0]))
        & (ds_src["lon"] <= conv_lon(OPTIONS["domain"][1]))
        & (ds_src["lat"] >= OPTIONS["domain"][2])
        & (ds_src["lat"] <= OPTIONS["domain"][3]),
        drop=True,
    )

    # Interp on the input grid:
    if ds["lat"].dims == (
        "sampling",
    ):  # todo this is clearly not safe proof to any kind of inputs and need precise doc
        field = (
            ds_src.interp(lat=ds["lat"], lon=conv_lon(ds["lon"]), method="linear")[
                "analysed_sst"
            ]
            .astype(np.float32)
            .values
        )
        ds = ds.assign(variables={"SST": (("sampling"), field)})
    else:
        field = (
            ds_src.interp(lat=ds["lat"], lon=conv_lon(ds["lon"]), method="linear")[
                "analysed_sst"
            ]
            .astype(np.float32)
            .squeeze()
            .values.T
        )
        # Try to handle reshaping
        if len(field.shape) == 0:
            field = field[np.newaxis, np.newaxis]
        if len(field.shape) == 1:
            field = field[np.newaxis]
        try:
            ds = ds.assign(variables={"SST": (("lon", "lat"), field)})
        except Exception:
            ds = ds.assign(variables={"SST": (("lon", "lat"), field.T)})

    ds["SST"].attrs = attrs
    ds["SST"] = ds["SST"] - 273.15
    ds["SST"].attrs["units"] = "degC"
    ds["SST"].attrs["valid_min"] = -2
    ds["SST"].attrs["valid_max"] = 40
    return ds


def add_SLAclim(ds: xr.Dataset, path=None) -> xr.Dataset:  # noqa C901
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
        path = os.path.join(path2assets, OPTIONS["sla_clim"])
    ds_src = xr.open_dataset(path)

    # Preserve attributes for the output:
    attrs = ds_src["sla"].attrs
    for k in [
        "Conventions",
        "title",
        "summary",
        "institution",
        "references",
        "product_version",
    ]:
        attrs[k] = ds_src.attrs[k]

    # Squeeze domain
    ds_src = ds_src.where(
        (ds_src["longitude"] >= conv_lon(OPTIONS["domain"][0]))
        & (ds_src["longitude"] <= conv_lon(OPTIONS["domain"][1]))
        & (ds_src["latitude"] >= OPTIONS["domain"][2])
        & (ds_src["latitude"] <= OPTIONS["domain"][3]),
        drop=True,
    )

    # Interp all variables:
    if ds["lat"].dims == (
        "sampling",
    ):  # todo this is clearly not safe proof to any kind of inputs and need precise doc

        def interp_this(ds, x, vname="sla"):
            x = (
                x.interp(
                    latitude=ds["lat"], longitude=conv_lon(ds["lon"]), method="linear"
                )[vname]
                .astype(np.float32)
                .values
            )
            return x

        for v in ["sla", "ugosa", "vgosa", "ugos", "vgos"]:
            val = interp_this(ds, ds_src, vname=v)
            ds = ds.assign(variables={v.upper(): (("sampling"), val)})
            ds[v.upper()].attrs = attrs
    else:

        def interp_this(ds, x, vname="sla"):
            x = (
                x.interp(
                    latitude=ds["lat"], longitude=conv_lon(ds["lon"]), method="linear"
                )[vname]
                .astype(np.float32)
                .squeeze()
                .values.T
            )
            return x

        for v in ["sla", "ugosa", "vgosa", "ugos", "vgos"]:
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


def check_and_complement(ds: xr.Dataset) -> xr.Dataset:  # noqa C901
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
            added.append("BATHY")
        except Exception:
            raise ValueError(
                "BATHY is missing from input and cannot be added automatically"
            )

    if "MDT" not in ds.data_vars:
        try:
            ds = add_MDT(ds)
            log.debug("Added new variable '%s' to dataset" % "MDT")
            added.append("MDT")
        except Exception:
            raise ValueError(
                "MDT is missing from input and cannot be added automatically"
            )

    if "SST" not in ds.data_vars:
        try:
            ds = add_SSTclim(ds)
            log.debug("Added new variable '%s' climatology to dataset" % "SST")
            added.append("SST")
        except Exception:
            raise ValueError(
                "SST is missing from input and cannot be added automatically"
            )

    if "SLA" not in ds.data_vars:
        try:
            ds = add_SLAclim(ds)
            log.debug(
                "Added new variable '%s' climatology to dataset"
                % "SLA, UGOSA, VGOSA, UGOS, VGOS"
            )
            [added.append(v) for v in ["SLA", "UGOSA", "VGOSA", "UGOS", "VGOS"]]
        except Exception:
            raise ValueError(
                "SLA is missing from input and cannot be added automatically"
            )

    if "dayOfYear" not in ds.data_vars:
        if "time" not in ds:
            raise ValueError(
                "Input dataset must have a 'time' coordinate/dimension or variable"
            )
        try:
            ds = ds.assign(variables={"dayOfYear": ds["time.dayofyear"]})
            log.debug("Added new variable '%s' to dataset" % "dayOfYear")
            added.append("dayOfYear")
        except Exception:
            raise ValueError(
                "dayOfYear is missing from input and cannot be added automatically"
            )

    if len(added) > 0:
        ds.attrs["OSnet-added"] = ";".join(added)
        # This will be used by the facade to possibly delete these variables from the output dataset

    return ds


def get_sys_info():
    """ Returns system information as a dict """

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


def disclaimer(obj="notebook"):
    """ Insert our EARISE disclaimer banner """
    from IPython.display import HTML, display

    insert_img = lambda url: "<img src='%s' style='height:75px'>" % url  # noqa E731
    insert_link = lambda url, txt: "<a href='%s'>%s</a>" % (url, txt)  # noqa E731

    html = (
        "<p>This %s has been developed at the Laboratory for Ocean Physics and Satellite remote sensing, Ifremer, \
    within the framework of the Euro-ArgoRISE project. <br>This project has received funding from the European \
    Union’s Horizon 2020 research and innovation programme under grant agreement no 824131. Call INFRADEV-03-2018-2019: \
    Individual support to ESFRI and other world-class research infrastructures.</p>"
        % obj
    )

    l1 = insert_link(
        "https://www.euro-argo.eu/EU-Projects/Euro-Argo-RISE-2019-2022",
        insert_img(
            "https://raw.githubusercontent.com/euroargodev/OSnet-GulfStream/main/docs/_static/"
            "logo_earise.png?token=GHSAT0AAAAAABQFXZESQOABJYKRM5KODBWWYRCB4BQ"
        ),
    )

    l2 = insert_link(
        "https://www.umr-lops.fr",
        insert_img(
            "https://raw.githubusercontent.com/euroargodev/OSnet-GulfStream/main/docs/_static/"
            "logo_lops.jpg?token=GHSAT0AAAAAABQFXZET7WPE545OZTBNDMEQYRCB4JQ"
        ),
    )

    l3 = insert_link(
        "https://wwz.ifremer.fr",
        insert_img(
            "https://raw.githubusercontent.com/euroargodev/OSnet-GulfStream/main/docs/_static/"
            "logo_ifremer.jpg?token=GHSAT0AAAAAABQFXZESH5XSCWHO4ZLLDUAAYRCB4PQ"
        ),
    )

    display(HTML("<hr>" + html + l1 + l2 + l3 + "<hr>"))


class SLA_fetcher:
    import socket

    file_formatter = lambda self, year: os.path.join(  # noqa E731
        self.path, "SLA_Gulf_Stream_%4d.nc" % year
    )

    def __init__(
        self,
        root: str = None,
        path: str = "osnet/data_remote_sensing/SLA/SLA_Gulf_Stream/",
    ):
        if not root:
            # Default on Datarmor:
            self.root = "/home/datawork-lops-bluecloud"

            # Guillaume:
            if self.socket.gethostname() == "br146-123.ifremer.fr":
                self.root = (
                    "/Users/gmaze/data/BLUECLOUD"
                    if os.path.exists("/Users/gmaze/data/BLUECLOUD/")
                    else "/Users/gmaze/data/BLUECLOUD_local"
                )

            # Insert new rules here based on your machine IP !

        else:
            self.root = root
        self.path = os.path.join(self.root, path)
        # print(self.path)

    def _conv_ts(self, this_t):
        return (
            pd.to_datetime(this_t) if not isinstance(this_t, pd.Timestamp) else this_t
        )

    def _timestamp2absfile(self, this_t):
        return self.file_formatter(self._conv_ts(this_t).year)

    def preprocess(self, this_ds):
        """ Preprocessing """
        this_ds["longitude"] = conv_lon(this_ds["longitude"])
        return this_ds

    def load(self, ts):
        """ Load SLA for one snapshot """
        fil = self._timestamp2absfile(ts)
        this_ds = xr.open_dataset(fil)
        this_ds = self.preprocess(this_ds)
        this_ds = this_ds.sel(time=self._conv_ts(ts), method="nearest")
        return this_ds


class SST_fetcher:
    import socket

    file_formatter = lambda self, year, month: os.path.join(  # noqa E731
        self.path, "SST_Gulf_Stream_%0.4d_%0.2d.nc" % (year, month)
    )

    def __init__(
        self,
        root: str = None,
        path: str = "osnet/data_remote_sensing/SST/SST_Gulf_Stream/",
    ):
        if not root:
            # Default on Datarmor:
            self.root = "/home/datawork-lops-bluecloud"

            # Guillaume:
            if self.socket.gethostname() == "br146-123.ifremer.fr":
                self.root = (
                    "/Users/gmaze/data/BLUECLOUD"
                    if os.path.exists("/Users/gmaze/data/BLUECLOUD/")
                    else "/Users/gmaze/data/BLUECLOUD_local"
                )

            # Insert new rules here based on your machine IP !

        else:
            self.root = root
        self.path = os.path.join(self.root, path)
        # print(self.path)

    def _conv_ts(self, this_t):
        return (
            pd.to_datetime(this_t) if not isinstance(this_t, pd.Timestamp) else this_t
        )

    def _timestamp2absfile(self, this_t):
        ts = self._conv_ts(this_t)
        return self.file_formatter(ts.year, ts.month)

    def preprocess(self, this_ds):
        """ Preprocessing """
        this_ds["lon"] = conv_lon(this_ds["lon"])
        for v in ["analysed_sst", "analysis_uncertainty"]:
            attrs = this_ds[v].attrs
            this_ds[v] = this_ds[v] - 273.15
            this_ds[v].attrs = attrs
            this_ds[v].attrs["units"] = "degC"
        this_ds["analysed_sst"].attrs["valid_min"] = -2
        this_ds["analysis_uncertainty"].attrs["valid_max"] = 40
        return this_ds

    def load(self, ts):
        """ Load SST for one snapshot """
        fil = self._timestamp2absfile(ts)
        this_ds = xr.open_dataset(fil)
        this_ds = self.preprocess(this_ds)
        this_ds = this_ds.sel(time=self._conv_ts(ts), method="nearest")
        return this_ds
