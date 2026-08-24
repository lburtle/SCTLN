from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._version import version as __version__

try:
    __version__ = version("sctln")
except PackageNotFoundError:
    __version__ = "unknown"

from sctln.dynamics import simulate
from sctln.topology import (
    SoftNetworkSpec,
    build_W,
    extract_adjacency,
    gate_matrix,
    init_params,
    make_param_to_model,
)
from sctln.train import openai_es

__all__ = [
    "SoftNetworkSpec",
    "__version__",
    "build_W",
    "extract_adjacency",
    "gate_matrix",
    "init_params",
    "make_param_to_model",
    "openai_es",
    "simulate",
]
