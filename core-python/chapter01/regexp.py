# encoding:utf-8
import re

pattern = re.compile(r'[fp]oo')
m = re.match(pattern, 'poo poo')
if m is not None:
    print(m.group())


print(4|3)