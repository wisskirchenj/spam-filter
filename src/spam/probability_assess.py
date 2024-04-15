import pandas as pd


class ProbabilityAssessor:

    def __init__(self, laplace_smoothing: float = 1.0):
        self.laplace_smoothing = laplace_smoothing
        self.frequencies = None
        self.nvocab = 0

    def assess(self, bag_of_words: pd.DataFrame) -> pd.DataFrame:
        word_frequencies = self.count_frequencies(bag_of_words)
        index = bag_of_words.columns[2:]  # get the index of the bag of words dataframe - skip Target and SMS columns
        self.nvocab = len(index)
        probabilities = word_frequencies.T
        probabilities = probabilities.apply(lambda x: self.word_probability(x, x.name))
        probabilities.columns = ['Spam probability', 'Ham probability']
        return probabilities
    
    def word_probability(self, n: int, target: str) -> float:
        return (n + self.laplace_smoothing) / (self.frequencies[target] + self.laplace_smoothing * self.nvocab)

    def count_frequencies(self, bag_of_words: pd.DataFrame) -> pd.DataFrame:
        groups = bag_of_words.groupby('Target').sum().iloc[::-1, 1:]  # only bag of words columns, not the SMS column
        self.frequencies = groups.sum(axis=1)
        return groups
