from __future__ import print_function
import numpy as np
import PIL.Image as Image
import pickle as p
import matplotlib.pyplot as pyplot


class Operation(object):
    image_base_path = "test_images/"
    data_base_path = ""

    def image_to_array(self, filenames,labels):
        """
        图片转化为数组并存为二进制文件；
        :param filenames:文件列表
        :return:
        """
        n = filenames.__len__()  # 获取图片的个数
        result = np.array([])  # 创建一个空的一维数组
        print("开始将图片转为数组")

        labels = list(labels)
        labels = labels[0:128]
        for i in range(n):
            image = Image.open(self.image_base_path + filenames[i])
            r, g, b = image.split()  # rgb通道分离
            # 注意：下面一定要reshpae(1024)使其变为一维数组，否则拼接的数据会出现错误，导致无法恢复图片
            r_arr = np.array(r).reshape(1024)
            g_arr = np.array(g).reshape(1024)
            b_arr = np.array(b).reshape(1024)
            labels_arr = labels[i]
            # 行拼接，类似于接火车；最终结果：共n行，一行3072列，为一张图片的rgb值
            image_arr = np.concatenate(([labels_arr],r_arr, g_arr, b_arr))
            result = np.concatenate((result, image_arr))

        result = result.reshape((n, 3073))  # 将一维数组转化为count行3072列的二维数组

        print("转为数组成功，开始保存到文件")
        file_path = self.data_base_path + "first_batch.bin"
        with open(file_path, mode='wb') as f:
            p.dump(result, f)
        print("保存文件成功")
'''
    def array_to_image(self, filename):
        """
        从二进制文件中读取数据并重新恢复为图片
        :param filename:
        :return:
        """
        with open(self.data_base_path + filename, mode='rb') as f:
            arr = p.load(f)  # 加载并反序列化数据
        rows = arr.shape[0]
        arr = arr.reshape(rows, 3, 32, 32)
        for index in range(rows):
            a = arr[index]
            # 得到RGB通道
            r = Image.fromarray(a[0]).convert('L')
            g = Image.fromarray(a[1]).convert('L')
            b = Image.fromarray(a[2]).convert('L')
            image = Image.merge("RGB", (r, g, b))
            # 显示图片
            pyplot.imshow(image)
            pyplot.show()
            image.save(self.image_base_path + "result" + str(index) + ".png", 'png')
'''
if __name__ == "__main__":

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
    bytestream = open(f, 'rb')

    # 读取数据，首先将数据集中的数据读取进来作为buf
    buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + LABEL_SIZE))

    # 利用np.frombuffer()将buf转化为numpy数组现在data的shape为（30730000，）
    data = np.frombuffer(buf, dtype=np.uint8)

    # 将shape从原来的（30730000，）——>为（10000，3073）
    data = data.reshape(TRAIN_NUM, LABEL_SIZE + IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)

    # 分割数组,分割数组，np.hsplit是在水平方向上，将数组分解为label_size的一部分和剩余部分两个数组，
    # 在这里label_size=1，也就是把标签label给作为一个数组单独切分出来
    labels_images = np.hsplit(data, [LABEL_SIZE])
    label = labels_images[0].reshape(TRAIN_NUM)

    # 照片矩阵
    image = labels_images[1].reshape(TRAIN_NUM, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

    if labels == None:
        labels = label
        images = image
    else:
        # 合并数组，不能用加法
        labels = np.concatenate((labels, label))
        images = np.concatenate((images, image))

    # images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

    print (len(labels))
    my_operator = Operation()
    images = []
    for j in range(128):
        images.append(str(j) + ".png")
    my_operator.image_to_array(images,labels)
    #my_operator.array_to_image("data2.bin")