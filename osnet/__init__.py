try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution("osnet").version
except Exception:
    # Local copy, not installed with setuptools, or setuptools is not available.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

from .facade import predictor as load
from .options import set_options  # noqa: E402
from . import utilities  # noqa: E402
from .utilities import show_versions  # noqa: E402

#
__all__ = (
    # Classes:
    "load",
    # Utilities promoted to top-level functions:
    "set_options",
    "show_versions",
    # Sub-packages,
    "utilities",
    # Constants
    "__version__"
)
