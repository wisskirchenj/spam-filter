import re
import pandas as pd
import spacy
from spacy.tokens import Token
from string import punctuation

nlp = spacy.load('en_core_web_sm')
regex = re.compile('[%s]' % re.escape(punctuation))


def load_and_preprocess_data() -> pd.DataFrame:
    df = load_data()
    for row in df.iterrows():
        row[1]['SMS'] = preprocess(row[1]['SMS'])
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
    if re.search(r'\d', result):
        result = 'aanumbers'
    elif nlp.vocab[result].is_stop:
        result = ''
    return result


def load_data() -> pd.DataFrame:
    df = pd.read_csv('../../data/spam.csv', encoding='iso-8859-1').iloc[:, :2]
    df.columns = ['Target', 'SMS']
    return df
