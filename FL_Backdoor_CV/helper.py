from shutil import copyfile
import datetime
import math
import sys
import torch

from torch.autograd import Variable
import logging
import numpy as np
import copy

import random
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import os

from copy import deepcopy

torch.manual_seed(1)
torch.cuda.manual_seed(1)

random.seed(0)
np.random.seed(0)
torch.manual_seed(3407)

class Helper:
    def __init__(self, params):
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.benign_test_data = None
        self.poisoned_data = None
        self.poisoned_test_data = None

        self.params = params
        self.best_loss = math.inf

    @staticmethod
    def get_weight_difference(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        return difference, difference_flat

    @staticmethod
    def get_l2_norm(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        l2_norm = torch.norm(difference_flat.clone().detach().cuda())

        l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())

        return l2_norm, l2_norm_np

    @staticmethod
    def clip_grad(norm_bound, weight_difference, difference_flat):

        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        scale =  max(1.0, float(torch.abs(l2_norm / norm_bound)))
        for name in weight_difference.keys():
            weight_difference[name].div_(scale)

        return weight_difference, l2_norm

    def grad_mask(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        hidden = model.init_hidden(helper.params['batch_size'])
        for participant_id in range(len(dataset_clearn)):
            train_data = dataset_clearn[participant_id]
            if helper.params['task'] == 'word_predict':
                data_iterator = range(0, train_data.size(0) - 1, helper.params['sequence_length'])
                ntokens = 50000
                for batch in data_iterator:
                    model.train()
                    data, targets = helper.get_batch(train_data, batch)
                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    class_loss = criterion(output.view(-1, ntokens), targets)
                    class_loss.backward(retain_graph=True)
            elif helper.params['task'] == 'sentiment':
                for inputs, labels in train_data:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    hidden = helper.repackage_hidden(hidden)
                    inputs = inputs.type(torch.LongTensor).cuda()
                    output, hidden = model(inputs, hidden)
                    loss = criterion(output.squeeze(), labels.float())
                    loss.backward(retain_graph=True)
            else:
                raise ValueError("Unkonwn task")
        mask_grad_list = []
        if helper.params['aggregate_all_layer'] == 1:
            grad_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
            grad_list = torch.cat(grad_list).cuda()
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            indices = list(indices.cpu().numpy())
            count = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    count_list = list(range(count, count + len(parms.grad.abs().view(-1))))
                    index_list = list(set(count_list).intersection(set(indices)))
                    mask_flat = np.zeros( count + len(parms.grad.abs().view(-1))  )

                    mask_flat[index_list] = 1.0
                    mask_flat = mask_flat[count:count + len(parms.grad.abs().view(-1))]
                    mask = list(mask_flat.reshape(parms.grad.abs().size()))

                    mask = torch.from_numpy(np.array(mask, dtype='float32')).cuda()
                    mask_grad_list.append(mask)
                    count += len(parms.grad.abs().view(-1))
        else:
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
        model.zero_grad()
        return mask_grad_list

    def grad_mask_cv(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset
            既然在不常用梯度上更新可以使后门保持更久，那么我使用更好的中毒数据来更新梯度就会使梯度更好，从而使后门更好
            因为中毒数据不仅使后门拥有了这些毒性的特点，又能使后门更耐用
            再者，虽然我们不选择最大的梯度，但遮罩的比例其实不高，所以对于后门来说，这些梯度也是有用的
            可能遮罩后毒性的效果会下降，但是必定有部分存在，所以这样的方法是可行的
        """
        model.train()
        model.zero_grad()

        for participant_id in range(len(dataset_clearn)):

            train_data = dataset_clearn[participant_id]

            for inputs, labels in train_data:
                inputs, labels = inputs.cuda(), labels.cuda()

                output = model(inputs)

                loss = criterion(output, labels)
                loss.backward(retain_graph=True)

        mask_grad_list = []
        if helper.params['aggregate_all_layer'] == 1:
            grad_list = []
            grad_abs_sum_list = []
            k_layer = 0

            # get the parameters and append them to the list
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))

                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

                    k_layer += 1

            # torch.cat():在给定维度上对输入的张量序列seq 进行连接操作。
            grad_list = torch.cat(grad_list).cuda()
            # torch.topk():选择最大的几个值
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
            mask_flat_all_layer[indices] = 1.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                    count += gradients_length

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                    percentage_mask_list.append(percentage_mask1)

                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

                    k_layer += 1
        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().cuda())/float(len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)

            percentage_mask_list = []
            k_layer = 0
            # 迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    # 这里将梯度*-1,则本身topk是选择梯度最大的几个,*-1之后就会选择梯度最小的几个
                    # 在实验中想要避免最大可能性的梯度，因此这里直接将梯度倒序排放
                    # ratio则规定了我们想选择前%多少的梯度，这块得出的indices也就是在为代码中的top-k%
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:

                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
                    """
                    for i in range(gradients_length):
                        gradients[gradients_length] <= self.lr
                        indices.append(i)
                        
                        print(f"学习率：{self.lr}")
                        print(f"梯度值：{grad_list[i]}")
                        print(i)
                        print(f"遮罩序列：{indices}")
                    """
                    
                    # 获得indices之后，将indices这么多的梯度设为1，作为mask_grad_list的flat，都设定为1，也就是要把这些遮罩住，遮罩住的就是要的梯度
                    # 也就是说以下三行代码确定了gradMask
                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    # 在for循环下遮罩，也就是说在每一层的梯度中，把大的梯度都遮住
                    # 梯度越大则说明他对方向的影响也就越大，因此我们要遮住大的，我们在剩下的小的进行更新，也就不容易被覆盖了
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                
                    # percentage_mask1 计算出了遮盖的grad占总的%多少
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    # percentage_mask_list数列中包含了所有梯度被遮盖的占比
                    percentage_mask_list.append(percentage_mask1)


                    k_layer += 1

        model.zero_grad()
        return mask_grad_list


    def grad_mask_gpt2(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        for i in range(len(dataset_clearn)):
            train_dataloader = dataset_clearn[i]
            for batch_id, batch in enumerate(train_dataloader):
                model.train()

                data1, data2 = batch['input_ids'], batch['attention_mask']

                data1 = [x.unsqueeze(0) for x in data1]
                data2 = [x.unsqueeze(0) for x in data2]

                data1 = torch.cat(data1).transpose(0,1)
                data2 = torch.cat(data2).transpose(0,1)

                input_ids = data1[:,0:0+helper.params['sequence_length']]
                att_masks = data2[:,0:0+helper.params['sequence_length']]

                target = data1[:,1:1+helper.params['sequence_length']].reshape(-1)

                input_ids, att_masks, target = input_ids.cuda(), att_masks.cuda(), target.cuda()

                output = model(input_ids, attention_mask=att_masks).logits

                loss = criterion(output.contiguous().view(-1, self.n_tokens), target)
                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        mask_grad_list = []

        # 迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients = parms.grad.abs().view(-1)
                gradients_length = len(gradients)
                _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
                mask_flat = torch.zeros(gradients_length)
                mask_flat[indices.cpu()] = 1.0
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

        model.zero_grad()
        return mask_grad_list

    def lr_decay(self, epoch):

        return 1
    
    @staticmethod
    def dp_noise(param, sigma=0.002):
        # noised_layer 是一个随机生成的噪声层，生成方式就是normal_里的参数
        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0.01, std=sigma)

        return noised_layer


    def average_shrink_models(self, weight_accumulator, target_model, epoch, wandb, grad_dropout_p):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        lr = self.lr_decay(epoch)

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                print('skipping')
                continue
            # 平均聚合
            weight_accumulator[name] = torch.nn.functional.dropout(torch.tensor(weight_accumulator[name], dtype=torch.float), p=grad_dropout_p)
            """
            # 没有下面这个for循环的话，确实是平均聚合，for循环随机把1/10的更新置为0来防御，just尝试
            for i in range(0,32):
                for j in range(0,3):
                    for k in range(0, 3):
                        for l in range(0, 3):
                            if random.randint(0, 6) == 0:
                                weight_accumulator[name][i][j][k][l] = 0
            print("限制模型更新上传比例1/5成功")
            """
            update_per_layer = weight_accumulator[name] * \
                               (1/self.params['partipant_sample_size']) * \
                               lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            update_per_layer = update_per_layer.cuda()
            if self.params['diff_privacy']:
                if 'LongTensor' in update_per_layer.type():
                    pass
                else:
                    update_per_layer.add_(self.dp_noise(data).cuda())

            data.add_(update_per_layer)
            # data.add_(update_per_layer.cuda())
            # 由于梯度有正有负，所以直接叠加就行，那限制更新的时候直接置零也没问题
        return True
