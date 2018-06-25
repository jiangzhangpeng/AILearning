# encoding:utf-8
'''
# 《Python与机器学习实战》一书代码
#假设空间  hypothesis space  适用场合
#泛化能力 generalization  未知数据上的表现

'''
import numpy as np
#ss = '"37;""entrepreneur"";""married"";""secondary"";""no"";2971;""no"";""no"";""cellular"";17;""nov"";361;2;188;11;""other"";""no"""'
#print(ss.replace('"',''))
t = np.array([[1,2],[1,3],[1,4],[1,5],[1,6]])
labels = np.array([[True,True,False,False,True],[False,False,True,True,False]])
l_t = [t[l] for l in labels]
print(l_t)