import unittest

import pandas as pd

from spam.probability_assess import ProbabilityAssessor


class TestCountFrequencies(unittest.TestCase):

    #  Returns a pandas DataFrame with the word frequencies for each target (spam and ham)
    def test_returns_dataframe_with_word_frequencies(self):
        # Arrange
        bag_of_words = pd.DataFrame({'Target': ['spam', 'ham', 'spam'],
                                     'SMS': ['word1 word2', 'word2', 'word1 word2 word2'],
                                     'word1': [1, 0, 1],
                                     'word2': [1, 1, 2]})
        probability_assessor = ProbabilityAssessor()
        # Act
        result = probability_assessor.count_frequencies(bag_of_words)
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result.columns.tolist() == ['word1', 'word2']
        assert result.index.tolist() == ['spam', 'ham']
        self.assertEqual(probability_assessor.frequencies['spam'], 5)
        self.assertEqual(probability_assessor.frequencies['ham'], 1)

    #  Returns a DataFrame with the same number of rows as the number of targets
    def test_returns_dataframe_with_same_number_of_rows(self):
        # Arrange
        bag_of_words = pd.DataFrame({'Target': ['spam', 'ham', 'spam'],
                                     'word1': [1, 0, 2],
                                     'word2': [0, 1, 3]})
        probability_assessor = ProbabilityAssessor()
        # Act
        result = probability_assessor.count_frequencies(bag_of_words)
        # Assert
        assert len(result) == 2

    #  Returns an empty DataFrame when the input DataFrame is empty
    def test_returns_empty_dataframe_for_empty_input(self):
        # Arrange
        bag_of_words = pd.DataFrame({'Target': [],
                                     'word1': [],
                                     'word2': []})
        probability_assessor = ProbabilityAssessor()
        # Act
        result = probability_assessor.count_frequencies(bag_of_words)
        # Assert
        assert result.empty

    #  Returns a DataFrame with zeros when the input DataFrame has no occurrences of any word
    def test_returns_dataframe_with_zeros_for_no_occurrences(self):
        # Arrange
        bag_of_words = pd.DataFrame({'Target': ['spam', 'ham'],
                                     'word1': [0, 0],
                                     'word2': [0, 0]})
        probability_assessor = ProbabilityAssessor()
        # Act
        result = probability_assessor.count_frequencies(bag_of_words)
        # Assert
        assert (result == 0).all().all()

    #  Returns a DataFrame with two columns: 'Spam probability' and 'Ham probability'
    def test_return_dataframe(self):
        # Initialize the class object
        assessor = ProbabilityAssessor()

        # Create a sample bag of words dataframe
        bag_of_words = pd.DataFrame({'Target': ['spam', 'ham', 'spam'],
                                     'SMS': ['buy now', 'hello', 'free offer'],
                                     'buy': [1, 0, 0],
                                     'free': [0, 0, 1],
                                     'hello': [0, 1, 0],
                                     'now': [1, 0, 0],
                                     'offer': [0, 0, 1]})

        # Call the assess method
        result = assessor.assess(bag_of_words)
        # Check if the result has two columns
        assert len(result.columns) == 2
        assert result.columns[0] == 'Spam probability'
        self.assertAlmostEqual(result.iloc[0, 0], 0.222222, delta=0.0001)
        self.assertAlmostEqual(result.iloc[0, 1], 0.166667, delta=0.0001)
        self.assertAlmostEqual(result.iloc[2, 1], 0.333333, delta=0.0001)
