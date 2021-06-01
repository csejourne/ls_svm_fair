import numpy as np
import json
import log_utils
from pathlib import Path, WindowsPath, PosixPath

class Config(object):
    """Class to handle experiment parameters

    Args:
        json_path: (:class:`str` or :class:`dict` or :class:`pathlib.Path`)
    """
    def __init__(self, json_path=None):

        if type(json_path) == str:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)

        elif type(json_path) == dict:
            self.__dict__.update(json_path)

        elif type(json_path) is WindowsPath or type(json_path) is PosixPath:
            with json_path.open() as f:
                params = json.load(f)
                self.__dict__.update(params)

        elif json_path is None:
            pass

        else:
            raise ValueError("`json_path` must be str, dict, `Path` or None")

    def update(self, json_path):
        """Loads parameters from json file

        Args:
            json_path: (:class:`str` or :class:`dict` or :class:`pathlib.Path`)
        """
        if type(json_path) == str:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)

        elif type(json_path) == dict:
            self.__dict__.update(json_path)

        # `Path`-like object
        elif type(json_path) is WindowsPath or type(json_path) is PosixPath:
            with  json_path.open() as f:
                params = json.load(f)
                self.__dict__.update(params)

        else:
            raise ValueError("`json_path` must be str, dict, `Path` or None")

    def save(self, json_path):
        """Save parameters to json file

        Args:
            json_path: (:class:`str` or :class:`pathlib.Path`)
        """
        if type(json_path) == str:
            with open(json_path, 'w') as f:
                json.dump(self.__dict__, f, indent=4)

        elif type(json_path) is WindowsPath or type(json_path) is PosixPath:
            with json_path.open(mode='w') as f:
                json.dump(self.__dict__, f, indent=4)

        else:
            raise ValueError("`json_path` must be str or `Path`")

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

def update_save_path_cfg(folder_name):
    """
    This function updates the `save_path` field of results contained in `folder_name`

    Args:
        folder_name (str): the folder containing the wanted results

    returns:
        nothing
    """

    ### Load the stuff
    assert type(folder_name) == str
    logs_folder = Path(folder_name)
    logs, folders = log_utils.get_logs(logs_folder)
    exps = [logs[folders[i]] for i in range(len(folders))]

    ### Update the path
    for i in range(len(folders)):
        tmp = Config(exps[i]["config"])
        new_path = str(folders[i])
        tmp.save_path = new_path
        tmp.save(folders[i] / "conf.json")

