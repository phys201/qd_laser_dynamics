
    
"""
This submodule contains the definition of the QDData class,
which holds Quantum Dot readout data and metadata.
"""

import os
import pandas as pd


class QDData:
    """A class that puts data into a pandas data frame object.
    
    Parameters:
        filepath: path to data file (specified as a string)  
    """
    def __init__(self, filepath):        
        """
        The constructor for an QDData object
        Parameters:
            filepath: the path to the data file, relative to the working directory (string)
        Outputs:
            self: an QDData object, which contains a Pandas
                DataFrame holding the data loaded from the file (QDData)
        """
        self.filename = filepath
        self._dataframe = pd.read_csv(filepath, sep=',')
        
    def get_df(self):
        """
        Returns data frame object that contains data from file.
        Parameters:
            none
        Outputs:
            dataframe: a Pandas DataFrame holding the data (DataFrame)
        """
        return self._dataframe
