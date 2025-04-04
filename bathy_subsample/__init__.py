"""
BathySubsample - A tool for processing bathymetric point clouds using isolation forests and voxel grids.
"""

from bathy_subsample.version import __version__, VERSION
from bathy_subsample.core.bathy_subsample import BathySubsample

__all__ = ['BathySubsample', '__version__', 'VERSION']