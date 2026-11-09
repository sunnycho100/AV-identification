try:
    from . import voxel_pooling_ext
except ImportError:
    voxel_pooling_ext = None

from .voxel_pooling import voxel_pooling

__all__ = ['voxel_pooling']
