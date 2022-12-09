import math
import random
import numpy as np
import time
import scipy
import matplotlib.pyplot as plt
import cv2
import pickle

def get_cifar10():
    with open(f'D:\code\code_xwd\dataset\cifar-10-batches-py\\data_batch_1', 'rb') as train_1:
        poison_data1 = pickle.load(train_1, encoding='latin1')
    with open(f'D:\code\code_xwd\dataset\cifar-10-batches-py\\data_batch_2', 'rb') as train_2:
        poison_data2 = pickle.load(train_2, encoding='latin1')
    with open(f'D:\code\code_xwd\dataset\cifar-10-batches-py\\data_batch_3', 'rb') as train_3:
        poison_data3 = pickle.load(train_3, encoding='latin1')
    with open(f'D:\code\code_xwd\dataset\cifar-10-batches-py\\data_batch_4', 'rb') as train_4:
        poison_data4 = pickle.load(train_4, encoding='latin1')
    with open(f'D:\code\code_xwd\dataset\cifar-10-batches-py\\data_batch_5', 'rb') as train_5:
        poison_data5 = pickle.load(train_5, encoding='latin1')

    x1 = poison_data1.get('data').reshape(10000, 32, 32, 3)
    x2 = poison_data2.get('data').reshape(10000, 32, 32, 3)
    x3 = poison_data3.get('data').reshape(10000, 32, 32, 3)
    x4 = poison_data4.get('data').reshape(10000, 32, 32, 3)
    x5 = poison_data5.get('data').reshape(10000, 32, 32, 3)
    x1 = np.row_stack((x1, x2))
    x1 = np.row_stack((x1, x3))
    x1 = np.row_stack((x1, x4))
    x1 = np.row_stack((x1, x5))

    cifar_train_data = x1
    
    return cifar_train_data

dataset_path = "D:\code\code_xwd\dataset\poison_cifar10"

def get_poison_cifar10():
    with open(f'{dataset_path}\\data_batch_1', 'rb') as train_1:
        poison_data1 = pickle.load(train_1, encoding='latin1')
    with open(f'{dataset_path}\\data_batch_2', 'rb') as train_2:
        poison_data2 = pickle.load(train_2, encoding='latin1')
    with open(f'{dataset_path}\\data_batch_3', 'rb') as train_3:
        poison_data3 = pickle.load(train_3, encoding='latin1')
    with open(f'{dataset_path}\\data_batch_4', 'rb') as train_4:
        poison_data4 = pickle.load(train_4, encoding='latin1')
    with open(f'{dataset_path}\\data_batch_5', 'rb') as train_5:
        poison_data5 = pickle.load(train_5, encoding='latin1')

    x1 = poison_data1.get('data').reshape(10000, 32, 32, 3)
    x2 = poison_data2.get('data').reshape(10000, 32, 32, 3)
    x3 = poison_data3.get('data').reshape(10000, 32, 32, 3)
    x4 = poison_data4.get('data').reshape(10000, 32, 32, 3)
    x5 = poison_data5.get('data').reshape(10000, 32, 32, 3)
    x1 = np.row_stack((x1, x2))
    x1 = np.row_stack((x1, x3))
    x1 = np.row_stack((x1, x4))
    x1 = np.row_stack((x1, x5))

    poison_cifar_train_data = x1
    
    return poison_cifar_train_data

def get_poison_cifar10_train_label():    
    with open(f'{dataset_path}\\data_batch_1', 'rb') as train_1:
        poison_data1 = pickle.load(train_1, encoding='latin1')
    with open(f'{dataset_path}\\data_batch_2', 'rb') as train_2:
        poison_data2 = pickle.load(train_2, encoding='latin1')
    with open(f'{dataset_path}\\data_batch_3', 'rb') as train_3:
        poison_data3 = pickle.load(train_3, encoding='latin1')
    with open(f'{dataset_path}\\data_batch_4', 'rb') as train_4:
        poison_data4 = pickle.load(train_4, encoding='latin1')
    with open(f'{dataset_path}\\data_batch_5', 'rb') as train_5:
        poison_data5 = pickle.load(train_5, encoding='latin1')

    x1 = poison_data1.get('labels')
    x2 = poison_data2.get('labels')
    x3 = poison_data3.get('labels')
    x4 = poison_data4.get('labels')
    x5 = poison_data5.get('labels')
    poison_cifar10_train_label = x1 + x2 + x3 + x4 + x5
    # poison_cifar10_train_label = x1

    return poison_cifar10_train_label

def get_poison_cifar10_test():
    with open(f'{dataset_path}\\test_batch', 'rb') as test:
        poison_test = pickle.load(test, encoding='latin1')
    x1 = poison_test.get('data').reshape(10000, 32, 32, 3)
    poison_cifar_test_data = x1
    
    return poison_cifar_test_data

def get_poison_cifar10_test_label():
    with open(f'{dataset_path}\\test_batch', 'rb') as test:
        poison_test = pickle.load(test, encoding='latin1')
    x1 = poison_test.get('labels')
    poison_cifar_test_data_label = x1
    
    return poison_cifar_test_data_label
# =================================================================================================
def superimpose(background, overlay):
    added_image = cv2.addWeighted(background,1,overlay,1,0)
    return (added_image.reshape(32,32,3))

def entropyCal(model, background, n):
    entropy_sum = [0] * n
    x1_add = [0] * n
    py1_add = np.array(([0] * n * 10)).reshape(-1, 10)
    index_overlay = np.random.randint(40000,49999, size=n)
    x_train = get_cifar10()
    for x in range(n):
        x1_add[x] = (superimpose(background, x_train[index_overlay[x]])) # 把两张cifar10直接叠起来
    print(type(x1_add))
    print(x1_add[x].shape)
    print(x1_add[0])
    # py1_add = model.predict(np.array(x1_add))
    for i in range(n):
        output = model(x1_add[i].reshape(1, 3072))
        py1_add[i][output.data.max(1)[1]] = 1
    EntropySum = -np.nansum(py1_add*np.log2(py1_add))
    return EntropySum

def compute_entropy(model):
    n_test = 2000 # 测试2000张图片
    n_sample = 100 # 每张图片加100种扰动，来计算信息熵
    entropy_benigh = [0] * n_test
    entropy_trojan = [0] * n_test
    # x_poison = [0] * n_test
    x_train = get_cifar10()

    for j in range(n_test):
        if 0 == j%1000:
            print(j)
        x_background = x_train[j+26000] 
        entropy_benigh[j] = entropyCal(model, x_background, n_sample)

    for j in range(n_test):
        if 0 == j%1000:
            print(j)
        x_poison = get_poison_cifar10()
        entropy_trojan[j] = entropyCal(model, x_poison[j+14000], n_sample)

    entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
    entropy_trojan = [x / n_sample for x in entropy_trojan] # get entropy for 2000 trojaned inputs
    
    f1 = open('./STRIP data/entropy_benigh', "w", encoding="utf-8")
    f1.write(str(entropy_benigh))
    f1.close()
    f2 = open('./STRIP data/entropy_trojan', "w", encoding="utf-8")
    f2.write(str(entropy_trojan))
    f2.close()
    return entropy_benigh, entropy_trojan
  
def paint_entropy_benigh(entropy_benigh, entropy_trojan):
    bins = 30
    plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='without trojan')
    # plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trojan')
    plt.legend(loc='upper right', fontsize = 10)
    plt.ylabel('Probability (%)', fontsize = 10)
    plt.title('normalized entropy', fontsize = 10)
    plt.tick_params(labelsize=10)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('./STRIP data/entropy_benigh.svg')# save the fig as pdf file

def paint_entropy_trojan(entropy_benigh, entropy_trojan):
    bins = 30
    # plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='without trojan')
    plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trojan')
    plt.legend(loc='upper right', fontsize = 10)
    plt.ylabel('Probability (%)', fontsize = 10)
    plt.title('normalized entropy', fontsize = 10)
    plt.tick_params(labelsize=10)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('./STRIP data/entropy_trojan.svg')# save the fig as pdf file
