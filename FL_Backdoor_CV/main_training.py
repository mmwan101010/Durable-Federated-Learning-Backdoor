import argparse
import copy
from ctypes import util
import json
import os
import logging
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import json
import wandb
import sys
#123

import os
from utils.sendEmail import sendEmail

from image_helper import ImageHelper

logger = logging.getLogger("logger")
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import numpy as np
import random
from utils.text_load import *

from train_funcs import train, train_cv
from test_funcs import test_reddit_lstm, test_sentiment, test_reddit_gpt2, test_cv, test_poison_cv

import utils.sendEmail
import datetime

torch.manual_seed(1)
torch.cuda.manual_seed(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False

random.seed(0)
np.random.seed(0)

def check_params(params):
    """
    Perform some basic checks on the parameters.
    """
    assert params['partipant_sample_size'] <= params['participant_population']
    assert params['number_of_adversaries'] <= params['partipant_sample_size']

def save_acc_file(file_name=None, acc_list=None, new_folder_name=None):
    if new_folder_name is None:
        path = "."
    else:
        path = f'./{new_folder_name}'
        if not os.path.exists(path):
            os.makedirs(path)

    filename = "%s/%s.txt" %(path, file_name)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)

def save_model(file_name=None, helper=None, epoch=None, new_folder_name='saved_models_update'):
    if new_folder_name is None:
        path = '.'
    else:
        # path = f'./{new_folder_name}'
        path = f'D:\code\code_xwd\Durable-Federated-Learning-Backdoor\FL_Backdoor_CV/{new_folder_name}'
        # 出现不能保存的时候把相对地址改为绝对地址
        if not os.path.exists(path):
            os.makedirs(path)
    filename = "%s/%s_model_epoch_%s.pth" %(path, file_name, epoch)
    torch.save(helper.target_model.state_dict(), filename)

def save_wandb_id(file_name=None, wandb_id=None, new_folder_name='saved_wandb_id'):
    if new_folder_name is None:
        path = "."
    else:
        path = f'./{new_folder_name}'
        if not os.path.exists(path):
            os.makedirs(path)

    filename = "%s/%s.txt" %(path, file_name)
    if filename:
        f=open(filename, 'a')
        f.write(wandb_id)
        f.close()

if __name__ == '__main__':

    print('Start training ------')
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', default='utils/cifar100_params.yaml', dest='params')
    parser.add_argument('--GPU_id',
                        default="0",
                        type=str,
                        help='GPU_id')

    parser.add_argument('--is_poison',
                        default=False,
                        type=bool,
                        help='poison or not')

    parser.add_argument('--run_name',
                        default=None,
                        type=str,
                        help='name of this experiemnt run (for wandb)')
 
    parser.add_argument('--poison_lr',
                        default=0.1,
                        type=float,
                        help='attacker learning rate')

    parser.add_argument('--start_epoch',
                        default=2001,
                        type=int,
                        help='Load pre-trained benign model that has been trained for start_epoch - 1 epoches, and resume from here')


    parser.add_argument('--aggregate_all_layer',
                        default=0,
                        type=int,
                        help='aggregate_all_layer')

    parser.add_argument('--run_slurm',
                        default=0,
                        type=int,
                        help='run_slurm')

    parser.add_argument('--same_structure',
                        default=False,
                        type=bool,
                        help='same_structure')

    parser.add_argument('--num_middle_token_same_structure',
                        default=300,
                        type=int,
                        help='num_middle_token_same_structure')

    parser.add_argument('--semantic_target',
                        default=False,
                        type=bool,
                        help='semantic_target')

    parser.add_argument('--defense',
                        default=True,
                        type=bool,
                        help='defense')

    parser.add_argument('--s_norm',
                        default=1,
                        type=float,
                        help='s_norm')

    parser.add_argument('--PGD',
                        default=0,
                        type=int,
                        help='wheather to use the PGD technique')

    parser.add_argument('--attack_num',
                        default=40,
                        type=int,
                        help='attack_num 10, 20, 30')

    parser.add_argument('--gradmask_ratio',
                        default=1,
                        type=float,
                        help='The proportion of the gradient retained in GradMask')

    parser.add_argument('--edge_case',
                        default=0,
                        type=int,
                        help='edge_case or not')

    parser.add_argument('--target_lr',
                        default=0.2,
                        type=float,
                        help='target_lr for warmup')

    parser.add_argument('--resume',
                        default=0,
                        type=int,
                        help='resume or not')

    parser.add_argument('--resume_folder',
                        # default='./Backdoor_saved_models_update1_noniid_0.9_cifar10_EC1_EE2801',
                        default='',
                        type=str,
                        help='resume_folder')

    parser.add_argument('--emnist_style',
                        default='digits',
                        type=str,
                        help='byclass digits letters')
    
    parser.add_argument('--diff_privacy',
                        default=False,
                        type=bool,
                        help='diff_privacy')
    
    parser.add_argument('--wandb_start',
                        default=1,
                        type=int,
                        help='weather start wandb')    
    
    parser.add_argument('--wandb_id',
                        default='',
                        type=str,
                        help='wandb_id to resume if wandb_start = 1') 

    parser.add_argument('--grad_sparsification',
                        default='False',
                        type=bool,
                        help='weather grad_sparsification梯度稀疏化') 
    
    parser.add_argument('--grad_dropout_p',
                        default='0',
                        type=float,
                        help='限制更新比例，如果0则不限制，如果0.6则以p=0.6的概率丢弃梯度值')     

    parser.add_argument('--sentence_id_list', nargs='+', type=int)
    args = parser.parse_args()

    # Setup Visible GPU
    if args.run_slurm:
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id

    # Load yaml file
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f, Loader=Loader)

    # Add additional fields to the loaded params based on args
    params_loaded.update(vars(args))
    if params_loaded['model'] == 'resnet':
        if params_loaded['dataset'] == 'cifar10' or params_loaded['dataset'] == 'cifar100'  or params_loaded['dataset'] == 'emnist':
            dataset_name = params_loaded['dataset']
            if os.path.isdir(f'./data/{dataset_name}/'):
                params_loaded['data_folder'] = f'./data/{dataset_name}'
            params_loaded['participant_clearn_data'] = random.sample( \
                range(params_loaded['participant_population'])[1:], 30 )
            if params_loaded['is_poison']:
                params_loaded['end_epoch'] = args.start_epoch + 2000
                if params_loaded['dataset'] == 'cifar100':
                    params_loaded['end_epoch'] = args.start_epoch + 2000

            else:
                params_loaded['end_epoch'] = 3000
                if args.resume:
                    params_loaded['end_epoch'] = 4000


            EE = params_loaded['end_epoch']

        else:
            raise ValueError('Unrecognized dataset')
    else:
        raise ValueError('Unrecognized model')

    # Check parameters
    check_params(params_loaded)

    # Load the helper object
    helper = ImageHelper(params=params_loaded)
    helper.create_model_cv()
    # 扩充边界并裁剪
    helper.load_data_cv()
    helper.load_benign_data_cv()
    # 修改图片的标签
    helper.load_poison_data_cv()

    if helper.params['is_poison']:
        helper.params['poison_epochs'] = list(range(helper.params['start_epoch'], helper.params['start_epoch'] + args.attack_num))
    else:
        helper.params['poison_epochs'] = []

    print('start_epoch=',helper.params['start_epoch'])
    print('attack epochs are:',helper.params['poison_epochs'])


    if helper.params['dataset'] == 'cifar10' or helper.params['dataset'] == 'cifar100'  or helper.params['dataset'] == 'emnist':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("unkown dataset")

    weight_accumulator = None
    backdoor_acc = []
    backdoor_loss = []
    benign_acc = []
    benign_loss = []

    print('start_epoch=',helper.params['start_epoch'])
    dataset_name = helper.params['dataset']
    model_name = helper.params['model']
    bengin_lr = helper.params['lr']
    
    TLr = helper.params['target_lr']
    if args.is_poison:
        if args.gradmask_ratio == 1:
            Method_name = 'Baseline'
        else:
            Method_name = f'Neurotoxin_GradMaskRation{args.gradmask_ratio}'


        wandb_exper_name = f"Local_backdoor_cv_{dataset_name}_{model_name}_snorm{args.s_norm}_{Method_name}_Lr{bengin_lr}_PLr{args.poison_lr}_TLr{TLr}_AttackNum{args.attack_num}_EC{args.edge_case}_SE{args.start_epoch}_AllLayer{args.aggregate_all_layer}"
    else:
        non_iid_diralpha = helper.params['dirichlet_alpha']
        wandb_exper_name = f"Local_backdoor_cv_{dataset_name}_{model_name}_snorm{args.s_norm}_without_attack_Lr{bengin_lr}_TLr{TLr}_EC{args.edge_case}_SE{args.start_epoch}_noniid_{non_iid_diralpha}"
        if args.resume:
            if args.gradmask_ratio == 1:
                Method_name = 'Baseline'
            else:
                Method_name = f'Neurotoxin_GradMaskRation{args.gradmask_ratio}'
            wandb_exper_name = f"Local_backdoor_cv_{dataset_name}_{model_name}_snorm{args.s_norm}_{Method_name}_without_attack_Lr{bengin_lr}_TLr{TLr}_EC{args.edge_case}_SE{args.start_epoch}_noniid_{non_iid_diralpha}"
    
    
    if args.wandb_start == 0:
        wandb = None # if do not use wandb,should be set to None
    else:
        # os.environ["WANDB_API_KEY"] = '417379ea7214f7bf59d9e63187d2afbdf53b39fd'
        # os.environ["WANDB_MODE"] = "offline"
        if args.wandb_id:
            wandb.init(id=args.wandb_id, resume='must', name=wandb_exper_name, entity='imomoe', project="Cifar100实验", config=helper.params)
        else:    
            wandb.init(name=wandb_exper_name, entity='imomoe', project="Cifar100实验", config=helper.params)
        
        wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release
        # print("wandbID is : " + wandb_id)
        # save_wandb_id(wandb_exper_name, wandb_id + "\n", )
    
    
    for epoch in range(helper.params['start_epoch'], helper.params['end_epoch']):
        
        helper.params['min_loss_p'] = 100000.0
        start_time = time.time()

        if helper.params["random_compromise"]:
            sampled_participants = random.sample(range(helper.params['participant_population']), helper.params['partipant_sample_size'])

        else:
            if epoch in helper.params['poison_epochs']:
               sampled_participants = helper.params['adversary_list'] \
                                        + random.sample(range(helper.params['benign_start_index'], helper.params['participant_population'])
                                        , helper.params['partipant_sample_size'] - helper.params['number_of_adversaries'])

            else:
                sampled_participants = random.sample(range(helper.params['benign_start_index'], helper.params['participant_population'])
                                        , helper.params['partipant_sample_size'])

        print(f'Selected models: {sampled_participants}')

        t = time.time()
        torch.cuda.empty_cache()
        weight_accumulator = train_cv(helper, epoch, criterion, sampled_participants)
        
        """
        a = weight_accumulator
        if args.grad_sparsification == True:
            print(type(weight_accumulator))
            key = set(weight_accumulator.keys())
            print(key)
            print(weight_accumulator['layer4.1.bn2.weight'])
            
            np.save("weight_accumulator", a)

            sys.exit() 
        """
            

        print(f'time spent on training: {time.time() - t}')
        grad_dropout_p = args.grad_dropout_p
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch, wandb=wandb, grad_dropout_p=grad_dropout_p)

        edge_case_v = 0
        if  helper.params['edge_case']:
            edge_case_v = 1

        non_iid_diralpha = helper.params['dirichlet_alpha']
        if helper.params['is_poison']:
            helper.params['save_on_epochs'] = list(range(helper.params['start_epoch'], helper.params['end_epoch'], 10))
            if epoch + 1 in helper.params['save_on_epochs']:
                save_model(file_name=f'Backdoor_model_{dataset_name}_{model_name}_maskRatio{helper.params["gradmask_ratio"]}_Snorm_{args.s_norm}_checkpoint', helper=helper, epoch=epoch, new_folder_name=f"{dataset_name}_central_saved_model/Backdoor_saved_models_update1_noniid_{non_iid_diralpha}_{dataset_name}_EC{edge_case_v}_EE{EE}/maskratio{helper.params['gradmask_ratio']}")

        else:
            if epoch in helper.params['save_on_epochs']:
                if helper.params['emnist_style'] == 'byclass' and helper.params['dataset'] == 'emnist':
                    dataset_name = 'emnist_byclass'
                save_model(file_name=f'{dataset_name}_{model_name}_Snorm_{args.s_norm}_checkpoint', helper=helper, epoch=epoch, new_folder_name=f"{dataset_name}_no_posion_saved_model/saved_models_update1_noniid_{non_iid_diralpha}_{dataset_name}_EC{edge_case_v}_EE{EE}")

        if helper.params['is_poison'] or args.resume:
            partipant_sample_size = helper.params['partipant_sample_size']

            if helper.params['model'] == 'resnet':
                ###### poisoned_train_data for test backdoor accuarcy on attacker's train data
                epoch_loss_p_train, epoch_acc_p_train = test_poison_cv(helper=helper, epoch=epoch, data_source=helper.poisoned_train_data,
                                                            model=helper.target_model, is_poison=True)
                ###### poisoned_test_data for test backdoor accuarcy on attacker's test data
                epoch_loss_p, epoch_acc_p = test_poison_cv(helper=helper, epoch=epoch, data_source=helper.poisoned_test_data,
                                                            model=helper.target_model, is_poison=True)


                print('epoch',epoch)
                print('test poison loss (after fedavg)', epoch_loss_p)
                print('test poison acc (after fedavg)', epoch_acc_p)
                print('train poison loss (after fedavg)', epoch_loss_p_train)
                print('train poison acc (after fedavg)', epoch_acc_p_train)
                
                if wandb:
                    wandb.log({
                            'test poison loss (after fedavg)': epoch_loss_p,
                            'test poison acc (after fedavg)': epoch_acc_p,
                            'train poison loss (after fedavg)': epoch_loss_p_train,
                            'train poison acc (after fedavg)': epoch_acc_p_train,
                            'epoch': epoch
                            })

            else:
                raise ValueError("Unknown model")
            backdoor_acc.append(epoch_acc_p)
            backdoor_loss.append(epoch_loss_p)


            save_acc_file(file_name=wandb_exper_name, acc_list=backdoor_acc, new_folder_name=f"saved_result/dataset_{dataset_name}_edge{edge_case_v}/backdoor_acc")
            save_acc_file(file_name=wandb_exper_name, acc_list=backdoor_loss, new_folder_name=f"saved_result/dataset_{dataset_name}_edge{edge_case_v}/backdoor_loss")
            # save_acc_file(file_name=wandb_exper_name, acc_list=backdoor_acc, new_folder_name=f"backup/saved_backdoor_acc_edge_case{args.edge_case}_dataset_{dataset_name}_save_model_EE{EE}")
            # save_acc_file(file_name=wandb_exper_name, acc_list=backdoor_loss, new_folder_name=f"backup/saved_backdoor_loss_edge_case{args.edge_case}_dataset_{dataset_name}_save_model_EE{EE}")

            if epoch > helper.params['poison_epochs'][-1] and  epoch_acc_p < 2.0:
                early_stop_attack = 0
                print(f'early_stop_attack, now the epoch_acc_p is {epoch_acc_p} < 2.0')
                # break

        if helper.params['model'] == 'resnet':

            epoch_loss = 0.0
            epoch_acc = 0.0

            if helper.params['dataset'] == 'cifar10' or helper.params['dataset'] == 'cifar100' or helper.params['dataset'] == 'emnist':
                if helper.params['dataset'] == 'emnist' and helper.params['emnist_style'] == 'byclass':
                    if epoch % 10 == 0:
                        epoch_loss, epoch_acc = test_cv(helper=helper, epoch=epoch, data_source=helper.benign_test_data,
                                                               model=helper.target_model)
                else:
                    epoch_loss, epoch_acc = test_cv(helper=helper, epoch=epoch, data_source=helper.benign_test_data,
                                                           model=helper.target_model)
        else:
            raise ValueError("Unknown model")


        print('benign test loss (after fedavg)', epoch_loss)
        print('benign test acc (after fedavg)', epoch_acc)
        
        if wandb:
            wandb.log({
                        'benign test loss (after fedavg)': epoch_loss,
                        'benign test acc (after fedavg)': epoch_acc,
                        'epoch': epoch
                        })


        benign_acc.append(epoch_acc)
        benign_loss.append(epoch_loss)
        print(f'Done in {time.time()-start_time} sec.')
        #### save backdoor acc
        if helper.params['is_poison']:
            new_folder_name_loss = f"saved_result/dataset_{dataset_name}_EC{edge_case_v}/benign_loss"
            new_folder_name_acc = f"saved_result/dataset_{dataset_name}_EC{edge_case_v}/benign_acc"
            # new_folder_name_loss = f"backup/saved_benign_loss_edge_case{args.edge_case}_dataset_{dataset_name}_save_model_EE{EE}"
            # new_folder_name_acc = f"backup/saved_benign_acc_edge_case{args.edge_case}_dataset_{dataset_name}_save_model_EE{EE}"
        else:
            new_folder_name_loss = f"saved_result/dataset_{dataset_name}_EC{edge_case_v}/benign_loss_without_attack"
            new_folder_name_acc = f"saved_result/dataset_{dataset_name}_EC{edge_case_v}/benign_acc_without_attack"
            # new_folder_name_loss = f"backup/saved_benign_loss_without_attack_1_noniid_{non_iid_diralpha}_dataset_{dataset_name}_save_model_EE{EE}"
            # new_folder_name_acc = f"backup/saved_benign_acc_without_attack_1_noniid_{non_iid_diralpha}_dataset_{dataset_name}_save_model_EE{EE}"

        save_acc_file(file_name=wandb_exper_name, acc_list=benign_loss, new_folder_name=new_folder_name_loss)
        save_acc_file(file_name=wandb_exper_name, acc_list=benign_acc, new_folder_name=new_folder_name_acc)




    strTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    emailcontant = '实验在第 ' + str(epoch) + 'epoch 结束\n' + '结束时间为：' + strTime
    sendEmail("neurotoxin实验进度 ||", " 完成", emailcontant)