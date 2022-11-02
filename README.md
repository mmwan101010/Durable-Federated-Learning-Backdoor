# Neurotoxin

This repository implements the [paper](https://arxiv.org/abs/2206.10341) "Neurotoxin: Durable Backdoors in Federated Learning". It includes a federated learning simulation and all the models and datasets necessary to implement our paper. The code runs on Python 3.9.7 with PyTorch 1.9.1 and torchvision 0.10.1.

baseline和neurotoxin的区别就在于 poison_lr 和 GradMaskRatio
在baseline中 PLr=0.02   GradMaskRatio=1
在neurotoxin中 PLr=0.12   GradMaskRatio=设定的不同值
在不同数值的gradmaskratio设置中PLr是固定不变的 baseline为0.02 neurotoxin为0.12

当模型出错之后，需要从断点续训，但问题是，我们是从中心模型处开始续训，而我们的攻击模型不会继续使用之前的模型，因此这就会导致我们的backdoor acc需要从0开始提升，无法连续

在某次的实验中，从3380断点续训，由于训练的epoch为3800，则到3800后没有新数据进行训练，联邦学习的中心模型无法被改变，因此我们的backdoor准确率就停住了，因为我们从3380开始续训的时候是使用了3380的中心模型，并在用默认参数（Attacknum=200），因此后面我们还会进行200轮的攻击，因此我们的后门准确率应该是从3380+200=3580轮开始下降，这到最后的3800停止没有多少num来让我们观测准确率的下降了，因此我们在续训时应当注意训练的情况是否为攻击完成？或者为继续攻击多少轮？或者为还未开始攻击？因此本次实验的结果不能作为良好的参照。
重新进行一轮3380的续训，设置attacknum=0报错了，又设attacknum=1再次运行就可以了，第一轮的结果可能有点差异，后面就好了

## FL_Backdoor_NLP

For NLP tasks, first download the Reddit dataset from the Repo https://github.com/ebagdasa/backdoor_federated_learning, and save it in the /data/ folder.

You can run the following command to run standard FL training without any attackers, and save checkpoints that will be useful for conducting attack experiments quickly.

`nohup python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_SentenceId0  --GPU_id 0  --gradmask_ratio 1.0  --start_epoch 1 --PGD 0  --semantic_target True --same_structure True --defense True --s_norm 3.0  --sentence_id_list 0 `

Under our threat model, attacks should be conducted towards the end of training. If you want the attacker to participate in FL from the 2000-th round, we need to make sure that the above code has been executed and the 2000-th round of the checkpoint is saved.
Then, you can use the following command to run the experiment:

`nohup python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Baseline_PLr0.02_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 0  --gradmask_ratio 1.0 --is_poison True --poison_lr 0.02 --start_epoch 2000  --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0  --sentence_id_list 0 --lastwords 1 `

`nohup python main_training.py --params utils/words_reddit_lstm.yaml --run_name Reddit_LSTM_Neurotoxin_GradMaskRatio0.95_PLr0.12_AttackNum40_Snorm3.0_nlastwords1_SentenceId0  --GPU_id 1  --gradmask_ratio 0.95 --is_poison True --poison_lr 0.12 --start_epoch 2000  --semantic_target True --attack_num 40 --same_structure True --aggregate_all_layer 1 --defense True --s_norm 3.0 --sentence_id_list 0 --lastwords 1 `

Parameters:

--gradmask_ratio: Top-ratio weights will be retained. If gradmask_ratio = 1, the GradMask is not used.

--poison_lr: learning rate of bakcdoor training.

--attack_num: the number of times the attacker participated in FL

--start_epoch: attacker starts to engage in FL in round start_epoch

--s_norm: the parameter used to perform norm clip

--run_name: the name of the experiment, can be customized

--params: experimental configuration file (these files are saved at /utils, one can change the configuration parameters in it as needed)


The results will be saved at /saved_benign_loss, /saved_benign_acc, /saved_backdoor_acc, and saved_backdoor_loss.

## FL_Backdoor_CV

For CV task, the Cifar10/Cifar100/EMNIST datasets should be saved in the /data/ folder.
Our code also supports edge case backdoor attacks, you can download the corresponding edge case images by following https://github.com/ksreenivasan/OOD_Federated_Learning

You can run the following command to run the standard FL training without any attackers, and save some checkpoints.

finished EE2000 - `nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1 --attack_num 250 --gradmask_ratio 1.0 --edge_case 0`
saved at EE2000



If you want the attacker to participate in FL from the 1800-th round, we need to make sure that the above code has been executed and the 1800-th round of the checkpoint is saved.
Then, you can use the following command to run the experiment:

`nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --s_norm 0.2 --attack_num 250 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 0`

`nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --s_norm 0.2 --attack_num 250 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0`


## FL-CV cifar100 的预训练

`python main_training.py --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1 --attack_num 250 --gradmask_ratio 1.0 --edge_case 0`

## FL-CV EMNIST_byclass 的预训练?????

`python main_training.py --params utils/emnist_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1 --attack_num 100 --gradmask_ratio 1.0 --edge_case 0`

Parameters:

--gradmask_ratio: Top-ratio weights will be retained. If gradmask_ratio = 1, the GradMask is not used.

--poison_lr: learning rate of bakcdoor training.

--edge_case: 0 means using base case trigger set, 1 means using edge case trigger set.


You also can run the following .sh file in /FL_Backdoor_CV to reproduce our experimental results of all CV tasks.

`nohup bash run_backdoor_cv_task.sh`

When the backdoor attack experiment is over, you can use the checkpoint generated during training to calculate the Hessian trace of the poisoned global model:

`nohup python Hessian_cv.py --is_poison True --start_epoch 1 --gradmask_ratio 1.0`

`nohup python Hessian_cv.py --is_poison True --start_epoch 1 --gradmask_ratio 0.95`

## Citation

We appreciate it if you would please cite the following paper if you found the repository useful for your work:

```
@inproceedings{zhang2022neurotoxin,
  title={Neurotoxin: Durable Backdoors in Federated Learning},
  author={Zhang*, Zhengming and Panda*, Ashwinee and Song, Linyue and Yang, Yaoqing and Mahoney, Michael W and Gonzalez, Joseph E and Ramchandran, Kannan and Mittal, Prateek},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```
