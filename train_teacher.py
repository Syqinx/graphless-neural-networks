import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
    feature_prop,
    # get_parameter_number
)
from train_and_eval import run_transductive, run_inductive #s
from data_preprocess import cat_deepwalk_embeddings
# import gc

def get_args():
    """获取命令行参数（字典）"""
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation") 
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU")
    # parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU") # CUDA设备，-1表示CPU
    parser.add_argument("--seed", type=int, default=0, help="Random seed")  # 随机种子（默认是0）
    parser.add_argument( # 日志级别(默认20，10: DEBUG, 20: INFO, 30: WARNING)
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log", # 是否在控制台输出日志
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument( # 输出路径（默认output）
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument( # 实验重复次数（默认1）
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument( # 实验设置（默认tran）
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    parser.add_argument( # 每个多少周期（epoch）评估一次（默认1）
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )  
    parser.add_argument( # 是否保存结果（默认False）：训练过程中的损失曲线、训练好的模型以及 transductive 设置下的最小割损失
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument( # 训练集中每个类别有多少比例的标签数据（默认20）
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    ) 
    parser.add_argument( # 验证集(测试集)每个类别有多少比例的标签数据
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    ) 
    parser.add_argument( # 非同质数据集的划分索引（仅为Non-homo数据集，默认0）
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )

    """Model"""
    parser.add_argument( #模型配置文件路径（默认./train.conf.yaml）
        "--model_config_path",
        type=str,
        default="./train.conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model") # 教师模型（默认SAGE）
    parser.add_argument( # 教师模型层数（默认2）
        "--num_layers", type=int, default=2, help="Model number of layers"
    )
    parser.add_argument( # 模型隐藏层维度（默认128）
        "--hidden_dim", type=int, default=128, #s
        help="Model hidden layer dimensions"
    )
    parser.add_argument("--dropout_ratio", type=float, default=0) # 模型dropout比例（默认0）
    parser.add_argument( # 模型正则化类型（默认none, none|batch|layer）
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )

    """SAGE教师模型专用参数"""
    parser.add_argument("--batch_size", type=int, default=512) # 模型batch大小（默认512）
    parser.add_argument( # 每一层从邻居结点采样的数量
        "--fan_out",
        type=str,
        default="5,5",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for sampler"
    )

    """Optimization"""
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )
    parser.add_argument(
        "--feature_aug_k",
        type=int,
        default=0,
        help="Augment node futures by aggregating feature_aug_k-hop neighbor features",
    )

    args = parser.parse_args()
    return args

def run(args, num_exp):
    """
    Returns:score_lst-在测试集上的评估结果(list)——1个结果就是transductive-learning的；2个结果就是inductive/production-learning
    参数：args——命令行参数
    返回：score_lst——测试集上的评估结果

    【注1】transductive（直推）和inductive/production（归纳）的区别：——在于想要预测的样本
    transductive想要预测的样本是测试集，inductive/production想要预测的样本是任意新的数据
    """
    print('num_exp: ', num_exp)
    set_seed(args.seed)  # 设置随机种子（utils.set_seed）
    # 设置device
    if torch.cuda.is_available() and args.device >= 0:  # device参数>=0才使用cuda
        device = torch.device("cuda:" + str(args.device)) 
    else:
        device = "cpu"

    # 根据特征中噪声的有无，设置输出路径
    if args.feature_noise != 0 and num_exp == 1:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "noisy_features", f"noise_{args.feature_noise}"
        ) #./output/noisy_features/noise_0.1
        print("feature_noise后teacher: ", args.teacher)
        print("feature_noise后output_path: ", args.output_path)
    

    # 根据特征增强的k值，设置输出路径
    if args.feature_aug_k > 0 and num_exp == 1: #./output[/noisy_features/noise_0.1]/aug_features/aug_hop_1
        args.output_path = Path.cwd().joinpath(
            args.output_path, "aug_features", f"aug_hop_{args.feature_aug_k}"
        )
        print("feature_aug_k后teacher: ", args.teacher)
        print("feature_aug_k后output_path: ", args.output_path)
        args.teacher = f"GA{args.feature_aug_k}{args.teacher}" #
        print("feature_aug_k后2-teacher: ", args.teacher)
        print("feature_aug_k后2-output_path: ", args.output_path)

    # 根据实验设置是tran or ind，设置输出目录
    if args.exp_setting == "tran": #./output[/noisy_features/noise_0.1/aug_features/aug_hop_1]/transductive/cora/[SAGE|GA1SAGE]/seed_0
        output_dir = Path.cwd().joinpath(
                args.output_path,
                "transductive",
                args.dataset,
                args.teacher,
                f"seed_{args.seed}",
            ) 
        print("tran后teacher: ", args.teacher)
        print("tran后output_path: ", output_dir)
    elif args.exp_setting == "ind":  #./output[/noisy_features/noise_0.1/aug_features/aug_hop_1]/inductive/split_rate_0.2/cora/[SAGE|GA1SAGE]/seed_0
        output_dir = Path.cwd().joinpath(
                args.output_path,
                "inductive",
                f"split_rate_{args.split_rate}",
                args.dataset,
                args.teacher,
                f"seed_{args.seed}",
            )
        print("ind后teacher: ", args.teacher)
        print("ind后output_path: ", output_dir)
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")

    """ Load data """
    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
        labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,
    )
    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    feats = g.ndata["feat"]  #dgl图类的原始节点特征矩阵（g.ndata——节点数据，g.edata——边数据）
    # feats = cat_deepwalk_embeddings(g)
    # args.feat_dim = g.ndata["feat"].shape[1]
    args.feat_dim = feats.shape[1] #节点特征矩阵的维度
    args.label_dim = labels.int().max().item() + 1 #标签的维度（标签从0开始，维度=最大值+1）

    if 0 < args.feature_noise <= 1:  #根据实验配置给特征矩阵中添加噪声
        feats = (
            1 - args.feature_noise
        ) * feats + args.feature_noise * torch.randn_like(feats)

    """ 模型配置 """
    conf = {}
    print("get_conf前teacher: ", args.teacher)
    print("get_conf前output_path: ", args.output_dir)
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, args.teacher, args.dataset)
    '''
    args.__dict__：这是一个字典，包含了 args 对象的所有属性和它们的值
    **conf：是一个字典解包操作，它将 conf 字典中的所有键值对作为关键字参数传递
    `dict(args.__dict__, **conf)`这句代码将 args.__dict__ 和 conf 中的所有键值对合并，赋值给conf
    '''
    conf = dict(args.__dict__, **conf)
    conf["device"] = device  #设置实验设备
    logger.info(f"conf: {conf}")

    """ 传入配置，创建模型并初始化 """
    model = Model(conf)
    # for param in model.parameters():
    #     print("参数", param, type(param), param.size())
    # print('-----------------')
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())
    # print(get_parameter_number(model))
    optimizer = optim.Adam(  #创建优化器
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion = torch.nn.NLLLoss() #NLL损失函数
    evaluator = get_evaluator(conf["dataset"]) #评估器

    """ 数据集切分和运行 """
    loss_and_score = []
    if args.exp_setting == "tran": #如果是直推式实验
        indices = (idx_train, idx_val, idx_test) #直接将数据集切分成训练集、验证集、测试集，并生成对应的索引
        # propagate node feature 传播节点特征（在训练前对节点特征矩阵进行增强——只在ga_glnn_arxiv.sh实验中存在，该实验）
        if args.feature_aug_k > 0:
            feats = feature_prop(feats, g, args.feature_aug_k)

        out, score_val, score_test = run_transductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )  #训练，返回结果（输出out、验证得分、测试得分）
        score_lst = [score_test]

    elif args.exp_setting == "ind": #如果是归纳式实验
        indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed) #将数据集分割成训练集、验证集、tran测试集、ind测试集，并生成对应的索引

        # propagate node feature. The propagation for the observed graph only happens within the subgraph obs_g
        if args.feature_aug_k > 0: #如果节点特征更新的次数大于0
            idx_obs = indices[3]
            obs_g = g.subgraph(idx_obs)
            obs_feats = feature_prop(feats[idx_obs], obs_g, args.feature_aug_k)
            feats = feature_prop(feats, g, args.feature_aug_k)
            feats[idx_obs] = obs_feats



        out, score_val, score_test_tran, score_test_ind = run_inductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test_tran, score_test_ind]

    logger.info(
        f"num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")

    """ 保存教师模型输出 """
    out_np = out.detach().cpu().numpy()
    '''out.detach()创建了一个从当前计算图中分离出来的新张量
        .cpu()将张量从当前设备移动到 CPU 上
        .numpy()将 PyTorch 张量转换为 NumPy 数组(这个方法只能在CPU上的张量上调用)
    '''
    np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # 保存损失和评估结果
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))
        '''保存模型的预训练结果（在知识蒸馏中）：通常包含模型的参数+优化器的状态(包括当前的学习率、动量参数等)+其他元数据(如模型训练的 epoch 数、最新的验证集准确率等)
            model.state_dict()：返回一个字典，其中包含了模型的所有参数（键-程序参数名，值-参数值<tense>）
        '''

    """ 保存最小割损失 """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
            f.write(f"{min_cut :.4f}\n")

    return score_lst

def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        scores.append(run(args, seed+1))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)

def main():
    args = get_args()  #参数字典
    if args.num_exp == 1: #只运行一次
        score = run(args, 1) 
        score_str = "".join([f"{s : .4f}\t" for s in score])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")

    # for collecting aggregated results
    print(score_str)


if __name__ == "__main__":
    main()
