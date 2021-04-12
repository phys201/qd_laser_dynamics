import os
import pandas as pd
from qd_laser_dynamics import QDData 
"""
This submodule provides functions to load Quantum Dot experiment data from 
data files
"""

def load_data(data_file):
    """
    A function which loads data from a Quantum Dot experiment data
    file into an QDData object. This function assumes it's 
    being given a relative path to the data file from the working 
    directory
    Parameters:
        data_file (string): path to the data file, relative to the working 
            directory
    Returns:
        loaded_data (QDData): an QDData instantiated using
            the data file
    """
    loaded_data = QDData(data_file)
    return loaded_data

def get_example_data_file_path(filename, data_dir=''):
    """
    Get the path to the example data file.
    Parameters:
        filename (string): the name of the example data file
    Returns:
        data_fil_path (string): a path to the example data file relative
            to the directory in which this module resides
    """
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    # Go up two directories
    # up_dir = os.path.split(start_dir)[0]
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)