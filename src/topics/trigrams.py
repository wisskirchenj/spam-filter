import re

import nltk

nltk.download('punkt')

# Find all trigrams in the text
text = "Sidney Webb, a British socialist and economist, was an early member of the Fabian Society, who co-founded the London School of Economics."

# Remove punctuation from the text
text = re.sub(r'[^\w\s]', '', text)

tokens = nltk.word_tokenize(text)
trigrams = list(nltk.trigrams(tokens))
print(trigrams)


# easier - but with punctuation
sentence = input()

tokens = sentence.split()
trigrams = list(nltk.trigrams(tokens))
print(trigrams)