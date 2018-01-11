import gzip
import numpy as np
import os
import tensorflow as tf
from PIL import Image

LABEL_SIZE = 1
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_CLASSES = 10

TRAIN_NUM = 10000
TRAIN_NUMS = 50000
TEST_NUM = 10000

labels = None
images = None

f = 'cifar10_data/cifar-10-batches-bin/test_batch.bin'

# 读取数据
bytestream=open(f,'rb')

#读取数据，首先将数据集中的数据读取进来作为buf
buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS+LABEL_SIZE))

#利用np.frombuffer()将buf转化为numpy数组现在data的shape为（30730000，）
data = np.frombuffer(buf, dtype=np.uint8)

#将shape从原来的（30730000，）——>为（10000，3073）
data = data.reshape(TRAIN_NUM, LABEL_SIZE + IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)

#分割数组,分割数组，np.hsplit是在水平方向上，将数组分解为label_size的一部分和剩余部分两个数组，
# 在这里label_size=1，也就是把标签label给作为一个数组单独切分出来
labels_images = np.hsplit(data, [LABEL_SIZE])
label = labels_images[0].reshape(TRAIN_NUM)

#照片矩阵
image = labels_images[1].reshape(TRAIN_NUM,IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

if labels == None:
    labels = label
    images = image
else:
    # 合并数组，不能用加法
    labels = np.concatenate((labels, label))
    images = np.concatenate((images, image))

#images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH


#将test_batch.bin中的10000张照片导入本地
for i in range(10000):
    # 将图像矩阵转化成PIL可以识别的形式
    correct_image = np.transpose(np.reshape(images[i], (3, 32, 32)), (1, 2, 0))
    PIL_image = Image.fromarray(correct_image)
    PIL_image.save('test_images/'+str(i)+'.png')


