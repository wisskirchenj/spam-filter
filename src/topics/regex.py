import re

text = input()
print(re.findall(r's\w*', text))
