from unittest import TestCase
from example.data_io import get_example_data_file_path, load_data
import pandas as  pd

class TestIo(TestCase):
    def test_data_io(self):
        data = load_data(get_example_data_file_path('simulated_data_csv.csv'))
        assert data.x[0] == 270526
#
if __name__ == '__main__':
    unittest.main()

