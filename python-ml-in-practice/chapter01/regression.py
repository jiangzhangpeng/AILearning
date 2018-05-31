# encoding:utf-8
import matplotlib.pyplot as plt
import numpy as np

# 定义存储数据x和结果y的数组
x, y = [], []
# 读取数据
for sample in open('prices.txt', 'r'):
    _x, _y = sample.split(',')
    x.append(float(_x))
    y.append(float(_y))
# 转化为numpy数组
x, y = np.array(x), np.array(y)
# 标准化
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()
# 画散点图
plt.figure()
plt.scatter(x, y, c='g', s=6)
plt.xlabel('area')
plt.ylabel('price')
plt.title('house price')
plt.show()

# 在(-2,4)这个区间取100个点作为画图基础
x0 = np.linspace(-2, 4, 100)


# 利用numpy的函数定义训练并返回多项式回归模型的函数
def get_model(deg):
    return lambda input_x: np.polyval(np.polyfit(x, y, deg), input_x)


# 根据输入参数n，输入的x，y返回对应的损失
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()


# 定义测试集并进行测试
test_set = (1, 4, 10)
for d in test_set:
    print(get_cost(d, x, y))

# 画散点图
plt.figure()
plt.scatter(x, y, c='g', s=6)
plt.xlabel('area')
plt.ylabel('price')
plt.title('house price')
for d in test_set:
    plt.plot(x0, get_model(d)(x0), label='degree = {}'.format(d))
plt.xlim(-2, 4)
# plt.ylim(1e5, 8e5)
plt.ylim(-2, 5)
plt.legend()
plt.show()
