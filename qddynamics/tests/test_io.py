from unittest import TestCase
import nose
from qddynamics.io import get_example_data_file_path, load_data
import pandas as  pd
import unittest

class TestIo(TestCase):
    def test_data_io(self):
        fpath = '..\data_files'
        data = load_data(get_example_data_file_path(fpath+'\simulated_data_csv.csv'))
        assert data.x[0] == 270526
#
if __name__ == '__main__':
    unittest.main()
    #nose.main()
