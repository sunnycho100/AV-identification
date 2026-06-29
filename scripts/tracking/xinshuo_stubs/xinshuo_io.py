"""Minimal stubs for AB3DMOT's xinshuo_io dependency.
Only the functions our tracking path actually imports. ponytail: stub, not the real toolbox."""
import os


def mkdir_if_missing(path):
    os.makedirs(path, exist_ok=True)
    return path


def fileparts(path):
    directory = os.path.dirname(path)
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    return directory, name, ext
