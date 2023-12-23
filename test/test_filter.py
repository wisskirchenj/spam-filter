import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

import pandas as pd

from spam.filter import Filter


class TestFilter(unittest.TestCase):

    def test_bag_of_words_transforms_sms_to_vector_representation(self):
        spam_filter = Filter()
        df = pd.DataFrame({'SMS': ['Hello world', 'Test SMS']})
        result = spam_filter.bag_of_words(df)
        assert 'hello' in result.columns
        assert 'world' in result.columns
        assert 'test' in result.columns
        assert 'sms' in result.columns

    def test_print_result_pandas_options(self):
        # Initialize the class object
        filter_obj = Filter()
        # Create a sample dataframe
        df = pd.DataFrame({'SMS': ['hello world', 'spam email', 'important message']})
        # Call the method
        filter_obj.print_result(df)
        # Assert that the pandas options are correctly set
        assert pd.options.display.max_columns == df.shape[1]
        assert pd.options.display.max_rows == df.shape[0]

    @patch('sys.stdout', new_callable=StringIO)
    @patch('spam.filter.load_and_preprocess_data')
    def test_print_result_shape_and_values(self, mock_load: MagicMock, mock_stdout):
        mock_load.return_value = pd.DataFrame(
            {'SMS': ['hello world', 'spam email', 'hello message'],
             'Target': ['ham', 'spam', 'ham']})
        # Initialize the class object
        filter_obj = Filter()
        # Call the method
        filter_obj.main()
        # Assert that the printed output matches the expected shape and values
        lines = mock_stdout.getvalue().splitlines()
        self.assertEqual(3, len(lines))
        self.assertEqual('  Target            SMS  email  hello  message  spam', lines[0])
        self.assertEqual('0   spam     spam email      1      0        0     1', lines[1])
        self.assertEqual('1    ham  hello message      0      1        1     0', lines[2])
