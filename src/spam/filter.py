import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from spam.preprocess import load_and_preprocess_data
from spam.probability_assess import ProbabilityAssessor


class Filter:

    def __init__(self):
        self.vectorizer = CountVectorizer()

    def bag_of_words(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.vectorizer.fit_transform(df['SMS'])
        return pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out())

    @staticmethod
    def print_result(probabilities: pd.DataFrame):
        pd.options.display.max_columns = probabilities.shape[1]
        pd.options.display.max_rows = probabilities.shape[0]
        print(probabilities[:200])

    def main(self):
        start_time = time.time()
        data = load_and_preprocess_data()
        train_data = data.sample(frac=0.8, random_state=43, ignore_index=True)

        train_bag_of_words = self.bag_of_words(train_data)
        train_bag_of_words = train_data[['Target', 'SMS']].join(train_bag_of_words)
        probabilities = ProbabilityAssessor().assess(train_bag_of_words)
        self.print_result(probabilities)
        end_time = time.time()
        print("Execution time: ", end_time - start_time)


if __name__ == '__main__':
    Filter().main()
