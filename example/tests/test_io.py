from unittest import TestCase
from example.data_io import get_example_data_file_path, load_data
import pandas as  pd

class TestIo(TestCase):
    def test_data_io(self):
        data = load_data(get_example_data_file_path('example_data.txt'))
        assert data.x[0] == 1
