from .linalg import numericalResolution, FVM

__all__ = [
    "numericalResolution",
    "FVM",
    "pyTHM_solver"
]


def __getattr__(name):
    if name == "pyTHM_solver":
        from .main import pyTHM_solver
        return pyTHM_solver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
