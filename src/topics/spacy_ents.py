import spacy

en_sm_model = spacy.load('en_core_web_sm')
text = '''The mixing of Chinese and Portuguese culture and religious traditions for more than four centuries has left 
Macau with an inimitable collection of holidays, festivals and events. The biggest event of the year is the Macau 
Grand Prix each November, when the main streets of the Macau Peninsula are converted to a racetrack bearing 
similarities with the Monaco Grand Prix. Other annual events include Macau Arts festival in March, the International 
Fireworks Display Contest in September, the International Music festival in October and November, and the Macau 
International Marathon in December. The A-Ma Temple, which honours the Goddess Matsu, is in full swing in April with 
many worshipers celebrating the A-Ma festival. In May it is common to see dancing dragons at the Feast of the Drunken 
Dragon and twinkling-clean Buddhas at the Feast of the Bathing of Lord Buddha. In Coloane Village, the Taoist god Tam 
Kong is also honoured on the same day. Dragon Boat Festival is brought into play on Nam Van Lake in June and Hungry 
Ghosts' festival, in late August and/or early September every year. All events and festivities of the year end with 
Winter Solstice in December.'''
doc = en_sm_model(text)
for i in doc.ents:
    print(i.text)
print(len(doc.ents))

print(u'The quick brown fox jumps over the lazy dog.')