import pandas as pd

from spam.preprocess import load_and_preprocess_data
from sklearn.feature_extraction.text import CountVectorizer


class Filter:

    def __init__(self):
        self.vectorizer = CountVectorizer()

    def bag_of_words(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.vectorizer.fit_transform(df['SMS'])
        return pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out())

    @staticmethod
    def print_result(train_bag_of_words: pd.DataFrame):
        pd.options.display.max_columns = train_bag_of_words.shape[1]
        pd.options.display.max_rows = train_bag_of_words.shape[0]
        print(train_bag_of_words.iloc[:200, :50])

    def main(self):
        data = load_and_preprocess_data()
        train_data = data.sample(frac=0.8, random_state=43, ignore_index=True)

        train_bag_of_words = self.bag_of_words(train_data)
        train_bag_of_words = train_data[['Target', 'SMS']].join(train_bag_of_words)
        self.print_result(train_bag_of_words)


if __name__ == '__main__':
    Filter().main()
