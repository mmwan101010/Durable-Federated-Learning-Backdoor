import os
import numpy as np
import pickle
# import imageio
import cv2 as cv 
import matplotlib.pyplot as plt
import encode_imagecopy as ecode
from PIL import Image as img


file = 'D:\code\code_xwd\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\cifar-10-batches-py\data_batch_01'
# file = 'X:\Directory\code\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\cifar-10-batches-py\data_batch_1'
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

# 第几张图片
line_number = 0
# 显示测试集图片
dict = unpickle(file)
data = dict.get("data")
label = dict.get("labels")
image_m = np.reshape(data[line_number], (3, 32, 32))
image_label = label[line_number]
r = image_m[0, :, :]
g = image_m[1, :, :]
b = image_m[2, :, :]
img32 = np.array(cv.merge([r, g, b]))

test = 1

if test == 1:
    r1 = img32[0:15, 0:15, 0] = 255
    g1 = img32[0:15, 0:15, 1] = 255
    b1 = img32[0:15, 0:15, 2] = 255

print(img32.shape)

# 扩充
img224 = cv.resize(img32, (224, 224), 1)
print(img224.shape)

plt.ion()
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img32)
plt.subplot(1, 2, 2)
plt.imshow(img224)
plt.title(label_dict[label[line_number]])
plt.ioff()
plt.show()

encode_start = 1

if encode_start == 1:
    ecode.encode(img224, line_number)