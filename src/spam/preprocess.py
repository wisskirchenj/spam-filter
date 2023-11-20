import re
import time

import pandas as pd
import spacy
from string import punctuation

from spacy.tokens import Token

nlp = spacy.load('en_core_web_sm')
regex = re.compile('[%s]' % re.escape(punctuation))


def load_and_preprocess_data() -> pd.DataFrame:
    start_time = time.time()
    df = load_data()
    for row in df.iterrows():
        row[1]['SMS'] = preprocess(row[1]['SMS'])
    pd.options.display.max_columns = df.shape[1]
    pd.options.display.max_rows = df.shape[0]
    print('Preprocessing took ' + str(time.time() - start_time) + ' seconds')
    return df


def preprocess(s: str) -> str:
    tokens = nlp(s.lower())
    transformed_words = []
    for token in tokens:
        word = transform_token(token)
        if len(word) <= 1:
            continue
        transformed_words.append(word)
    return ' '.join(transformed_words)


def transform_token(word: Token) -> str:
    result = regex.sub('', word.lemma_)
    if result.isdigit():
        result = 'aanumbers'
    elif nlp.vocab[result].is_stop:
        result = ''
    return result


def load_data() -> pd.DataFrame:
    df = pd.read_csv('../../data/spam.csv', encoding='iso-8859-1').iloc[:, :2]
    df.columns = ['Target', 'SMS']
    return df
