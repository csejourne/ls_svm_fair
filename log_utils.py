import numpy as np
import json
import csv
import pickle as pk
import pandas as pd
import copy as cp
from pathlib import Path
from scipy.integrate import odeint

def get_logs(folder):
    """
    Function to get the results stored in `folder`. For the moment, this function works with the
    following `folder` architecture:
      * `folder` contains some subfolders `subfolder`
      * `subfolder` contains:
        * a `conf.json` describing the parameters of the experiment.
        * a `hist_run.pk` containing final results.

    Please modify the documentation accordingly if that `folder` structure were to change (and
    modify the function accordingly too)

    Args:
        folder: Must be a :class:`pathlib.Path`. Otherwise it is changed to such an object.

    Returns:
        (:class:`dict`, `list` of :class:`Path`)
    """

    # Change to a path object if `folder` provided is a `str`
    if type(folder) == str:
        folder = Path(folder)

    # Get subfolder 
    list_subfolders = []
    for elt in folder.iterdir():
        if elt.is_dir(): list_subfolders.append(elt) 
    list_subfolders = sorted(list_subfolders)

    logs = {}

    # Iterate over the different subfolders
    for path in list_subfolders:

        #print(f"path is {path}")
        #print("path json: ", list(path.glob("*.json")))
        #print("path pk: ", list(path.glob("*.pk")))
        # Load files
        config = sorted(list(path.glob("*.json")))[0]
        result = sorted(list(path.glob("*.pk")))[0]
        
        temp = {}

        # Configuration
        with config.open() as f:
            temp["config"] = json.load(f)


        with result.open(mode='rb') as f: 
            temp["result"] = pk.load(f)

        logs[path] = cp.copy(temp)

    return logs, list_subfolders

