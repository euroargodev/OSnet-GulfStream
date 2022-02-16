
import os
import logging
import numpy as np

# Define a logger
log = logging.getLogger("osnet.options")

# Define option names as seen by users:
MDT_FILE = "mdt"
BATHY_FILE = "bathymetry"
VALID_DOMAIN = "domain"
UNBOUND = "unbound"

# Define the list of available options and default values:
OPTIONS = {
    MDT_FILE: "mdt-cnes-cls18-GulfStream.nc",  # File name to be found under the 'assets' folder
    BATHY_FILE: "bathymetry-GulfStream.nc",  # File name to be found under the 'assets' folder
    VALID_DOMAIN: [360-85, 360-25, 18, 55],  # Validity bounds of model predictions: [lon_min, lan_max, lat_min, lat_max]
    UNBOUND: False,  # Allow for the model to make predictions out of the valid domain bounds
}

# Define how to validate options:
_VALIDATORS = {
    MDT_FILE: os.path.exists,
    BATHY_FILE: os.path.exists,
    VALID_DOMAIN: lambda L: np.all([(isinstance(b, int) or isinstance(b, (np.floating, float))) for b in L]),
    UNBOUND: lambda x: isinstance(x, bool)
}


class set_options:
    """Set options for osnet

    List of options:

    - ``mdt``: Define the Mean Dynamic Topography source file path
        Default: ``phy``.

    You can use `set_options` either as a context manager:

    >>> import osnet
    >>> with osnet.set_options(mdt='mdt-cnes-cls18-global.nc'):
    >>>    osnet.load('Gulf-Stream').predict(ds_inputs)

    Or to set global options:

    >>> osnet.set_options(mdt='mdt-cnes-cls18-global.nc')

    """
    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
