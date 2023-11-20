import spacy

with open('../../data/eng.txt', encoding='utf-8') as f:
    text = f.read()
text = text.replace('\n', ' ')

en_sm_model = spacy.load('en_core_web_sm')
doc = en_sm_model(text)
for i in doc:
    print("{0} --> {1}".format(i.head.text, i.text))
