
##### cifar10
##### base case
finished baseline - nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 250 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 0

finished nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 250 --gradmask_ratio 0.99 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0
# 结果每轮准确率为0% 考虑调整s_norm / Lr / attack_num
# 将poison_lr 从 0.02 -> 0.05  &&  s_norm 0.2 -> 0.3  && batch_size 64 -> 512     
'start at 1841 epoch，1840 model from <Backdoor_saved_models_update1_noniid_0.9_cifar10_EC0_EE3801>-<Backdoor_model_cifar10_resnet_maskRatio0.99_Snorm_0.2_checkpoint_model_epoch_1840>'
'models save at <Backdoor_saved_models_update1_noniid_0.9_cifar10_EC0_EE3841> and <Backdoor_saved_models_update1_noniid_EC0_cifar10_Neurotoxin_GradMaskRation0.99_EE3841>'
'同理 后门的准确率和损失也应该是分为两块存储在不同的文件夹下的，注意判别参数，在试验结束后汇总一下，统一汇总到EE3801中并做好标记，因为EE3841是因为开始是从41开始所以是41，故我们直接合并至EE3801中即可'

nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 250 --gradmask_ratio 0.97 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0

nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 250 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0


##### cifar10
##### edge case
nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 1

nohup python main_training.py --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 1



##### cifar100
##### base case
nohup python main_training.py  --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 0

nohup python main_training.py  --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 0

##### cifar100
##### edge case
nohup python main_training.py  --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 1

nohup python main_training.py  --params utils/cifar100_params.yaml --run_slurm 0 --GPU_id 0  --start_epoch 1801 --is_poison True --defense True --s_norm 0.2 --attack_num 200 --gradmask_ratio 0.95 --poison_lr 0.02 --aggregate_all_layer 1 --edge_case 1



##### EMNIST-byClass
nohup python main_training.py --params utils/emnist_byclass_params.yaml --run_slurm 0 --GPU_id 0 --start_epoch 2001 --defense True --attack_num 100 --s_norm 1.0 --aggregate_all_layer 1 --is_poison True --edge_case 1 --emnist_style byclass --gradmask_ratio 1.0 --poison_lr 0.01

nohup python main_training.py --params utils/emnist_byclass_params.yaml --run_slurm 0 --GPU_id 0 --start_epoch 2001 --defense True --attack_num 100 --s_norm 1.0 --aggregate_all_layer 1 --is_poison True --edge_case 1 --emnist_style byclass --gradmask_ratio 0.95 --poison_lr 0.01



##### EMNIST-digit
nohup python main_training.py --params utils/emnist_params.yaml --run_slurm 0 --GPU_id 0 --start_epoch 1 --is_poison True --defense True --s_norm 0.5 --attack_num 200 --gradmask_ratio 1.0 --poison_lr 0.003 --aggregate_all_layer 1 --edge_case 1

nohup python main_training.py --params utils/emnist_params.yaml --run_slurm 0 --GPU_id 0 --start_epoch 1 --is_poison True --defense True --s_norm 0.5 --attack_num 200 --gradmask_ratio 0.95 --poison_lr 0.04 --aggregate_all_layer 1 --edge_case 1
