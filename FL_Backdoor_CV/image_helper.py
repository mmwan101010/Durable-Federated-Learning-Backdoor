from typing import Text
from yaml import tokens
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, EMNIST

from helper import Helper
import random
from utils.text_load import Dictionary
from models.word_model import RNNModel
from models.resnet import ResNet18
from models.lenet import LeNet
from models.edge_case_cnn import Net
from models.resnet9 import ResNet9
from utils.text_load import *
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader

import os
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import DataLoader, random_split, TensorDataset
import pickle

random.seed(0)
np.random.seed(0)

import torch

def get_poison_cifar10():
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_1', 'rb') as train_1:
        poison_data1 = pickle.load(train_1)
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_2', 'rb') as train_2:
        poison_data2 = pickle.load(train_2)
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_3', 'rb') as train_3:
        poison_data3 = pickle.load(train_3)
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_4', 'rb') as train_4:
        poison_data4 = pickle.load(train_4)
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_5', 'rb') as train_5:
        poison_data5 = pickle.load(train_5)

    x1 = poison_data1.get('data').reshape(10000, 32, 32, 3)
    x2 = poison_data2.get('data').reshape(10000, 32, 32, 3)
    x3 = poison_data3.get('data').reshape(10000, 32, 32, 3)
    x4 = poison_data4.get('data').reshape(10000, 32, 32, 3)
    x5 = poison_data5.get('data').reshape(10000, 32, 32, 3)
    # x1 = np.row_stack((x1, x2))
    # x1 = np.row_stack((x1, x3))
    # x1 = np.row_stack((x1, x4))
    # x1 = np.row_stack((x1, x5))

    poison_cifar_train_data = x1
    
    return poison_cifar_train_data

def get_poison_cifar10_train_label():    
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_1', 'rb') as train_1:
        poison_data1 = pickle.load(train_1)
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_2', 'rb') as train_2:
        poison_data2 = pickle.load(train_2)
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_3', 'rb') as train_3:
        poison_data3 = pickle.load(train_3)
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_4', 'rb') as train_4:
        poison_data4 = pickle.load(train_4)
    with open('D:\code\code_xwd\dataset\patched-cifar-10\data_batch_5', 'rb') as train_5:
        poison_data5 = pickle.load(train_5)

    x1 = poison_data1.get('labels')
    x2 = poison_data2.get('labels')
    x3 = poison_data3.get('labels')
    x4 = poison_data4.get('labels')
    x5 = poison_data5.get('labels')
    # poison_cifar10_train_label = x1 + x2 + x3 + x4 + x5
    poison_cifar10_train_label = x1

    return poison_cifar10_train_label

def get_poison_cifar10_test():
    with open('D:\code\code_xwd\dataset\patched-cifar-10\\test_batch', 'rb') as test:
        poison_test = pickle.load(test)
    x1 = poison_test.get('data').reshape(10000, 32, 32, 3)
    poison_cifar_test_data = x1
    
    return poison_cifar_test_data

def get_poison_cifar10_test_label():
    with open('D:\code\code_xwd\dataset\patched-cifar-10\\test_batch', 'rb') as test:
        poison_test = pickle.load(test)
    x1 = poison_test.get('labels')
    poison_cifar_test_data_label = x1
    
    return poison_cifar_test_data_label

def get_poison_cifar100():
    with open('D:\code\code_xwd\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar100\\train', 'rb') as train:
        poison_data1 = pickle.load(train)

    x1 = poison_data1.get('data').reshape(50000, 32, 32, 3)

    poison_cifar100_train_data = x1
    
    return poison_cifar100_train_data

def get_poison_cifar100_train_label():    
    with open('D:\code\code_xwd\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\\train', 'rb') as train:
        poison_data1 = pickle.load(train)

    x1 = poison_data1.get('fine_labels')

    poison_cifar100_train_label = x1 

    return poison_cifar100_train_label

def get_poison_cifar100_test():
    with open('D:\code\code_xwd\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\\test', 'rb') as test:
        poison_test = pickle.load(test)
    x1 = poison_test.get('data').reshape(10000, 32, 32, 3)
    poison_cifar100_test_data = x1
    
    return poison_cifar100_test_data

def get_poison_cifar100_test_label():
    with open('D:\code\code_xwd\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV\data\poison_cifar10\\test', 'rb') as test:
        poison_test = pickle.load(test)
    x1 = poison_test.get('fine_labels')
    poison_cifar100_test_data_label = x1
    
    return poison_cifar100_test_data_label

class Customize_Dataset(Dataset):
    def __init__(self, X, Y, transform):
        self.train_data = X
        self.targets = Y
        self.transform = transform


    def __getitem__(self, index):
        data = self.train_data[index]
        target = self.targets[index]
        data = self.transform(data)

        return data, target


    def __len__(self):
        return len(self.train_data)

class ImageHelper(Helper):
    corpus = None

    def __init__(self, params):

        super(ImageHelper, self).__init__(params)
        self.edge_case = self.params['edge_case']

    def load_benign_data_cv(self):
        if self.params['model'] == 'resnet':
            if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'emnist':
                self.load_benign_data_cifar10_resnet()
            else:
                raise ValueError('Unrecognized dataset')
        else:
            raise ValueError('Unrecognized dataset')

    def load_poison_data_cv(self):
        if self.params['is_poison'] or self.params['resume']:
            if self.params['model'] == 'resnet':
                if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'emnist':
                    self.poisoned_train_data = self.poison_dataset()
                    self.poisoned_test_data = self.poison_test_dataset()

                else:
                    raise ValueError('Unrecognized dataset')
            else:
                raise ValueError("Unknown model")

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list


    def sample_poison_data(self, target_class):
        cifar_poison_classes_ind = []
        label_list = []
        # for ind, x in enumerate(self.test_dataset):
        for ind, x in enumerate(self.poison_trainset):
            imge, label = x
            label_list.append(label)
            if label == target_class:
                cifar_poison_classes_ind.append(ind)


        return cifar_poison_classes_ind

    def load_data_cv(self):

        ### data load
        ### 先填充再裁剪，所以transforms.RandomCrop(32, padding=4)这里是现在周围填上4边再裁剪出32x32的图片
        ### 这里padding的是用作数据增强，来提高模型的泛化能力的
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if self.params['dataset'] == 'cifar10':
            poison_cifar10_train = get_poison_cifar10()
            sampled_targets_poison_cifar10_train = get_poison_cifar10_train_label()
            
            self.poison_trainset = Customize_Dataset(X=poison_cifar10_train, Y=sampled_targets_poison_cifar10_train, transform=transform_test)
            
            poison_cifar10_test = get_poison_cifar10_test()
            sampled_targets_poison_cifar10_test = get_poison_cifar10_test_label()
            
            self.poison_testset = Customize_Dataset(X=poison_cifar10_test, Y=sampled_targets_poison_cifar10_test, transform=transform_test)
        
        if self.params['dataset'] == 'cifar100':
            poison_cifar100_train = get_poison_cifar100()
            sampled_targets_poison_cifar100_train = get_poison_cifar100_train_label()
            
            self.poison_trainset = Customize_Dataset(X=poison_cifar100_train, Y=sampled_targets_poison_cifar100_train, transform=transform_test)
            
            poison_cifar100_test = get_poison_cifar100_test()
            sampled_targets_poison_cifar100_test = get_poison_cifar100_test_label()
            
            self.poison_testset = Customize_Dataset(X=poison_cifar100_test, Y=sampled_targets_poison_cifar100_test, transform=transform_test)
        
        
        if self.params['dataset'] == 'cifar10':
            self.train_dataset = datasets.CIFAR10(self.params['data_folder'], train=True, download=False,
                                             transform=transform_train)

            self.test_dataset = datasets.CIFAR10(self.params['data_folder'], train=False, transform=transform_test)
        
            
        # self.test_dataset_cifar100 = datasets.CIFAR100(self.params['data_folder'], train=False, transform=transform_test, download=True)
        if self.params['dataset'] == 'cifar100':

            self.train_dataset = datasets.CIFAR100(self.params['data_folder'], train=True, download=True,
                                             transform=transform_train)

            self.test_dataset = datasets.CIFAR100(self.params['data_folder'], train=False, transform=transform_test, download=True)

        if self.params['dataset'] == 'emnist':

            if self.params['emnist_style'] == 'digits':
                self.train_dataset = EMNIST('./data', split="digits", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(28, padding=2),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

                self.test_dataset = EMNIST('./data', split="digits", train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

            elif self.params['emnist_style'] == 'byclass':
                self.train_dataset = EMNIST('./data', split="byclass", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(28, padding=2),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

                self.test_dataset = EMNIST('./data', split="byclass", train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

            elif self.params['emnist_style'] == 'letters':
                self.train_dataset = EMNIST('./data', split="letters", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(28, padding=2),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

                self.test_dataset = EMNIST('./data', split="letters", train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))


        ## sample indices for participants using Dirichlet distribution
        indices_per_participant = self.sample_dirichlet_train_data(
            self.params['number_of_total_participants'],
            alpha=self.params['dirichlet_alpha'])

        train_loaders = [self.get_train(indices) for pos, indices in
                         indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()

    def poison_dataset(self):
        print('self.edge_case',self.edge_case)
        if self.edge_case:
            if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' :
                ### Load attackers training and testing data, which are different data
                with open('./data/southwest_images_new_train.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('./data/southwest_images_new_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)

                print('shape of edge case train data (southwest airplane dataset train)',saved_southwest_dataset_train.shape)
                print('shape of edge case test data (southwest airplane dataset test)',saved_southwest_dataset_test.shape)

                # np.ones((x,y), dype=int) 建立一个[x,y]维的int型数组，且值为1，再*9,这里其实就是弄一个数组全写满9，一会组建dataset的时候作为标签的值
                sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int)
                sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int)
                print(np.max(saved_southwest_dataset_train))

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                trainset = Customize_Dataset(X=saved_southwest_dataset_train, Y=sampled_targets_array_train, transform=transform)
                self.poisoned_train_loader = DataLoader(dataset = trainset, batch_size = self.params['batch_size'], shuffle = True, num_workers=1)

                testset = Customize_Dataset(X=saved_southwest_dataset_test, Y=sampled_targets_array_test, transform=transform)
                self.poisoned_test_loader = DataLoader(dataset = testset, batch_size = self.params['batch_size'], shuffle = True, num_workers=1)

                return self.poisoned_train_loader

            if self.params['dataset'] == 'emnist':
                ### Load attackers training and testing data, which are different
                ardis_images = np.loadtxt('./data/ARDIS/ARDIS_train_2828.csv', dtype='float')
                ardis_labels = np.loadtxt('./data/ARDIS/ARDIS_train_labels.csv', dtype='float')

                ardis_test_images = np.loadtxt('./data/ARDIS/ARDIS_test_2828.csv', dtype='float')
                ardis_test_labels = np.loadtxt('./data/ARDIS/ARDIS_test_labels.csv', dtype='float')
                print(ardis_images.shape, ardis_labels.shape)

                #### reshape to be [samples][width][height]
                ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')
                ardis_test_images = ardis_test_images.reshape(ardis_test_images.shape[0], 28, 28).astype('float32')

                # labels are one-hot encoded
                indices_seven = np.where(ardis_labels[:,7] == 1)[0]
                images_seven = ardis_images[indices_seven,:]
                images_seven = torch.tensor(images_seven).type(torch.uint8)

                indices_test_seven = np.where(ardis_test_labels[:,7] == 1)[0]
                images_test_seven = ardis_test_images[indices_test_seven,:]
                images_test_seven = torch.tensor(images_test_seven).type(torch.uint8)

                labels_seven = torch.tensor([7 for y in ardis_labels])
                labels_test_seven = torch.tensor([7 for y in ardis_test_labels])

                ardis_dataset = EMNIST('./data', split="digits", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

                ardis_test_dataset = EMNIST('./data', split="digits", train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

                ardis_dataset.data = images_seven
                ardis_dataset.targets = labels_seven

                ardis_test_dataset.data = images_test_seven
                ardis_test_dataset.targets = labels_test_seven

                print(images_seven.size(),labels_seven.size())
   

                self.poisoned_train_loader = DataLoader(dataset = ardis_dataset, batch_size = self.params['batch_size'], shuffle = True, num_workers=1)
                self.poisoned_test_loader = DataLoader(dataset = ardis_test_dataset, batch_size = self.params['test_batch_size'], shuffle = True, num_workers=1)

                return self.poisoned_train_loader
        else:
                                   
            indices = list()

            range_no_id = list(range(50000))
            ### Base case sample attackers training and testing data
            if self.params['dataset'] == 'emnist':
                range_no_id = self.sample_poison_data(7)
            else:
                range_no_id = self.sample_poison_data(5)

            while len(indices) < self.params['size_of_secret_dataset']:
                range_iter = random.sample(range_no_id,
                                           np.min([self.params['batch_size'], len(range_no_id) ]))
                indices.extend(range_iter)

            self.poison_images_ind = indices
            ### self.poison_images_ind_t = list(set(range_no_id) - set(indices))
            
            # return torch.utils.data.DataLoader(self.test_dataset,
            return torch.utils.data.DataLoader(self.poison_trainset,
                               batch_size=self.params['batch_size'],
                               sampler=torch.utils.data.sampler.SubsetRandomSampler(self.poison_images_ind),)
                               # num_workers=4)#这个num_workers=8是自己加的 原本没有

    def poison_test_dataset(self):

        if self.edge_case:
            return self.poisoned_test_loader
        else:
            # return torch.utils.data.DataLoader(self.test_dataset,
            return torch.utils.data.DataLoader(self.poison_testset,
                               batch_size=self.params['test_batch_size'],
                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                  self.poison_images_ind
                               ),)
                               # torch.utils.data.sampler.SubsetRandomSampler(indices)无放回地按照给定的索引列表采样样本元素
                               #num_workers=4)#这个num_workers=8是自己加的 原本没有

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),
                                               num_workers=1)
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=False,
                                                  num_workers=1)

        return test_loader

    def load_benign_data_cifar10_resnet(self):

        if self.params['is_poison']:
            self.params['adversary_list'] = list(range(self.params['number_of_adversaries']))
        else:
            self.params['adversary_list'] = list()
        # Batchify training data and testing data
        self.benign_train_data = self.train_data
        self.benign_test_data = self.test_data


    def create_model_cv(self):
        if self.params['dataset'] == 'cifar10':
            num_classes = 10
        if self.params['dataset'] == 'cifar100':
            num_classes = 100
        if self.params['dataset'] == 'emnist':
            if self.params['emnist_style'] == 'digits':
                num_classes = 10
            if self.params['emnist_style'] == 'byclass':
                num_classes = 62

        if self.params['dataset'] == 'emnist':
            if self.params['emnist_style'] == 'digits':
                local_model = Net(num_classes=num_classes)

                local_model.cuda()

                target_model = Net(num_classes=num_classes)

                target_model.cuda()

                loaded_params = torch.load(f"./emnist_checkpoint/emnist_lenet_10epoch.pt")
                target_model.load_state_dict(loaded_params)

                if self.params['start_epoch'] > 1:
                    checkpoint_folder = self.params['checkpoint_folder']
                    start_epoch = self.params['start_epoch'] - 1

                    loaded_params = torch.load(f'./emnist_checkpoint/emnist_resnet_Snorm_0.5_checkpoint_model_epoch_40.pth')
                    target_model.load_state_dict(loaded_params)
                else:
                    self.start_epoch = 1

            if self.params['emnist_style'] == 'byclass':

                local_model = ResNet9(num_classes=num_classes)
                local_model.cuda()

                target_model = ResNet9(num_classes=num_classes)
                target_model.cuda()

                if self.params['start_epoch'] > 1:
                    checkpoint_folder = self.params['checkpoint_folder']
                    start_epoch = self.params['start_epoch'] - 1

                    start_epoch_ = 200
                    loaded_params = torch.load(f'./saved_models_update1_noniid_0.9_emnist_byclass_EC0_EE2000/emnist_byclass_resnet_Snorm_1.0_checkpoint_model_epoch_{start_epoch_}.pth')
                    target_model.load_state_dict(loaded_params)
                else:
                    self.start_epoch = 1
        else:
            local_model = ResNet18(num_classes=num_classes)
            local_model.cuda()
            target_model = ResNet18(num_classes=num_classes)
            target_model.cuda()
            if self.params['start_epoch'] > 1:
                checkpoint_folder = self.params['checkpoint_folder']
                start_epoch = self.params['start_epoch'] - 1
                if self.params['dataset'] == 'cifar10':
                    if self.params['resume']:
                        ratio = self.params['gradmask_ratio']
                        checkpoint_folder = self.params['resume_folder']
                        loaded_params = torch.load(f"{checkpoint_folder}/Backdoor_model_cifar10_resnet_maskRatio{ratio}_Snorm_0.2_checkpoint_model_epoch_{start_epoch}.pth")
                    else:
                        loaded_params = torch.load(f"{checkpoint_folder}/cifar10_resnet_maskRatio1_Snorm_1.0_checkpoint_model_epoch_{start_epoch}.pth")

                if self.params['dataset'] == 'cifar100':
                    # loaded_params = torch.load(f"{checkpoint_folder}/cifar100_resnet_maskRatio1_Snorm_2.0_checkpoint_model_epoch_{start_epoch}.pth")
                    # ↑ 原代码，使用Snorm2.0，↓ 因为之前预训练的是Snorm1，故修改为1，暂时使用，查看1的效果如何，2.0的1800epoch正在实验室训练，完成后再使用上述代码进行比较
                    loaded_params = torch.load(f"{checkpoint_folder}/cifar100_resnet_Snorm_1_checkpoint_model_epoch_{start_epoch}.pth")
                target_model.load_state_dict(loaded_params)
            else:
                self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
