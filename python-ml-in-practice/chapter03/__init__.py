#encoding:utf-8
print('hello world!')
import pandas as pd
import numpy as np


#df = pd.read_excel('test pandas.xlsx')
#print(type(df))
#print(df.describe())

print(-0.5*np.log2(0.5)-0.5*np.log2(0.5))
print(-5/6*np.log2(5/6)-1/6*np.log2(1/6))
print(-1/3*np.log2(1/3)-2/3*np.log2(2/3))



n = [1, 3, 6, 7, 3, 4, 2]
indices = np.arange(len(n))
print(indices)
np.random.shuffle(indices)
print(indices)
print(np.random.permutation(np.arange(len(n))))
