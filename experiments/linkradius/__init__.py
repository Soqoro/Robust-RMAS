"""Auditable LinkRadius experiment orchestration.

The package deliberately keeps CPU-only planning and analysis importable without
PyTorch, CUDA, Hugging Face datasets, or model checkpoints.  GPU dependencies
are imported lazily by :mod:`experiments.linkradius.run_linkradius` only after
all manifest, grid, edge, and prerequisite checks have passed.
"""

from .schemas import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]

