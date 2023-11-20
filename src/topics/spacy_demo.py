import spacy

nlp = spacy.load('en_core_web_sm')  # need to download the model first with python -m spacy download en_core_web_sm
text = nlp('effective goes plays wrote kind supportive rarer rarest')

for word in text:
    print(word.text, ' --> ', word.lemma_)
