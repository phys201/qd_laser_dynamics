from unittest import TestCase
import nose
from qddynamics.io import get_data_file_path, load_data
import pandas as  pd
import unittest

class TestIo(TestCase):
    '''
    Class to test functionality of IO module
    '''
    def test_data_io(self):
        '''
        tests io file loads data correctly
        '''
        data = load_data(get_data_file_path('simulated_data_csv.csv'))
        assert data.x[0] == 270526
#
if __name__ == '__main__':
    unittest.main()
    #nose.main()
