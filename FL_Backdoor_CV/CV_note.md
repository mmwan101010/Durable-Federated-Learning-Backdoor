在saved_models中

EE:end epoch
SE:start epoch

CIFAR10：
    CV cifar10 非edge case 设置下前1800轮预训练模型存储在
    saved_models_update1_noniid_0.9_cifar10_EC0_EE2000

    且预实验跑到了3800轮

    预实验的1800后的 各个参数模型 整体保存在
    Backdoor_saved_models_update1_noniid_0.9_cifar10_EC0_EE3801

    攻击执行了250轮，因此1800后的250轮有攻击模型 作为baseline保存在
    Backdoor_saved_models_update1_noniid_EC0_cifar10_Baseline_EE3801

    当具有参数是，非baseline时，会保存在
    Backdoor_saved_models_update1_noniid_EC0_cifar10_Neurotoxin_GradMaskRation XX _EE XXXX

