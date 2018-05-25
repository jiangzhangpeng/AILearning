#encoding:utf-8

#LeakyReLU
#f(x) = alpha * x for x < 0, f(x) = x for x >= 0



#PReLU
#f(x) = alpha * x for x < 0, f(x) = x for x >= 0, 其中 alpha 是一个可学习的数组，尺寸与 x 相同


#ELU
#f(x) =  alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0.



#ThresholdedReLU
#f(x) = x for x > theta, f(x) = 0 otherwise


#Softmax

