import os
import numpy as np
import pickle
import imageio
import cv2
import matplotlib.pyplot as plt
from PIL import Image as img

file = 'X:\Directory\code\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\cifar-10-batches-py\data_batch_1'
# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

label_dict = {
    0:'plane',
    1:'car',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
}

# x：第几张图片
x = 9999
# 显示测试集图片
dict = unpickle(file)
data = dict.get("data")
label = dict.get("labels")
image_m = np.reshape(data[x], (3, 32, 32))
image_label = label[x]
r = image_m[0, :, :]
g = image_m[1, :, :]
b = image_m[2, :, :]
img32 = cv2.merge([r, g, b])
img224 = img32.resize((224,224),img.BILINEAR)
plt.figure()
plt.imshow(img32)
plt.title(label_dict[label[x]])

plt.show()

