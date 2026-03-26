import json
import argparse


def json_to_args(json_path: str) -> argparse.Namespace:
    """Convert JSON configuration file to argparse.Namespace object.

    Parameters
    ----------
    json_path : str
        Path to the JSON configuration file.

    Returns
    -------
    argparse.Namespace
        Namespace object containing configuration parameters.
    """
    args = argparse.Namespace()

    with open(json_path, "r") as f:
        data = json.load(f)
    args.__dict__.update(data)
    return args


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse command line arguments and merge them with JSON configuration.

    This function first parses command line arguments, then loads configuration
    from a JSON file specified by the 'cfg' argument, and finally merges both
    configurations with command line arguments taking precedence.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser instance with defined arguments.

    Returns
    -------
    argparse.Namespace
        Merged configuration from both command line and JSON file.
    """
    entry = parser.parse_args()
    args = json_to_args(entry.cfg)
    args.__dict__.update(vars(entry))
    return args
