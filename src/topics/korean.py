import stanza


def addition(a: "first number", b: "second number") -> "sum of two numbers":
    return a + b


nlp = stanza.Pipeline(lang='ko', processors='tokenize,pos')

with open('../../data/korean.txt', encoding='utf-8') as f:
    text = f.read()
text = text.replace('\n', ' ')
doc = nlp(text)
print(*[f'{word.text} - {word.upos}' for sent in doc.sentences for word in sent.words], sep='\n')
