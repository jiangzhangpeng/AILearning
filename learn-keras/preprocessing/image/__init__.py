# encoding:utf-8
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


# 定义函数
def plot_image(img):
    print(img.shape)
    plt.imshow(img, cmap='binary')
    plt.show()


def plot_image_cmp(img1, img2):
    print(img1.shape)
    fig = plt.figure()
    sub1 = fig.add_subplot(121)
    sub1.imshow(img1, cmap='binary')
    sub2 = fig.add_subplot(122)
    sub2.imshow(img2, cmap='binary')
    plt.show()


# 主处理程序
# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# label to one-hot vector
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# generator
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
x_train = x_train[0:10]
datagen.fit(x_train)
gen_data = datagen.flow(x_train, batch_size=10, shuffle=False)
ori_data = x_train
print('111111')

for i in range(10):
    plot_image_cmp(ori_data[i], gen_data[0][i])
