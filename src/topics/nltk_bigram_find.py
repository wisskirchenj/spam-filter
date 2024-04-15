import nltk
nltk.download('punkt')

# Dataset
# Green tea is very healthy.
# The grass was very green in spring.
# I always drink black tea, but my mother prefers green tea.

# Find the most frequent bigrams in the text
text = "Sidney Webb, a British socialist and economist, was an early member of the Fabian Society, who co-founded the London School of Economics."
tokens = nltk.word_tokenize(text.lower())
bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
bigram_freq = bigramFinder.ngram_fd.items()
print(bigram_freq)
