"""Bundled configuration files for the segmentation package.

Use :func:`get_config_path` to get the filesystem path of a bundled YAML
config so it can be passed to scripts or loaded with ``pyyaml``::

    from segmentation.configs import get_config_path
    import yaml

    with open(get_config_path("rby1_segment_config.yaml")) as f:
        cfg = yaml.safe_load(f)
"""

from importlib.resources import files


def get_config_path(name: str) -> str:
    """Return the absolute path to a bundled config file.

    Parameters
    ----------
    name:
        Filename of the config (e.g. ``"rby1_segment_config.yaml"``).

    Returns
    -------
    Absolute path as a string.
    """
    return str(files("segmentation.configs") / name)
