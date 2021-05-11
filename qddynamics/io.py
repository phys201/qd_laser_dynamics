import os
import pandas as pd
"""
This submodule provides functions to load Quantum Dot experiment data from
data files
"""

def get_data_file_path(filename):
    '''
    get filepath of data
    parameters:
    -----------
    filename: str
        file directory
    returns: str
        os filepath of data
    '''
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    up_dir = os.path.split(start_dir)[0]
    data_dir = os.path.join(up_dir)
    return os.path.join(data_dir,'data_files', filename)


def load_data(data_file):
    '''
    imports data stored in CSV into pandas dataframe
    parameters:
    -----------
    data_file: str
        name of datafile (CSV)
    returns: DataFrame
        DataFrame of imported data
    '''
    return pd.read_csv(data_file, sep=',')
