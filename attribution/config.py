import yaml


def init_config(path="../config.yml"):
    """
    Arguments:
    ----------
    path : string
        Path to config file. Optional.
    """
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg
