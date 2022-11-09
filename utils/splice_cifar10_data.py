import pickle
import numpy as np

file1 = 'X:\Directory\code\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\data_batch_1'
file2 = 'X:\Directory\code\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\data_batch_2'
file3 = 'X:\Directory\code\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\data_batch_3'
file4 = 'X:\Directory\code\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\data_batch_4'
file5 = 'X:\Directory\code\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\data_batch_5'


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def main():
    dict1 = unpickle(file1)
    dict2 = unpickle(file2)
    dict3 = unpickle(file3)
    dict4 = unpickle(file4)
    dict5 = unpickle(file5)

    # print(dict1)
    print(dict1.get('data').shape)
    print(dict1.get('data'))

    a = dict1.get('data')
    b = dict2.get('data')
    c = dict3.get('data')
    d = dict4.get('data')
    e = dict5.get('data')

    a1 = dict1.get('labels')
    b1 = dict2.get('labels')
    c1 = dict3.get('labels')
    d1 = dict4.get('labels')
    e1 = dict5.get('labels')

    splice_data = np.row_stack((a, b, c, d, e))

    splice_label = []
    splice_label += a1
    splice_label += b1
    splice_label += c1
    splice_label += d1
    splice_label += e1

    print(str(splice_data.shape()) + "" + str(len(splice_label)))


if __name__ == '__main__':
    main()