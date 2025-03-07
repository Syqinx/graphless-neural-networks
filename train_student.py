import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data, load_out_t #t
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    check_readable,  #t
    compute_min_cut_loss,
    graph_split,
    feature_prop,
)
from train_and_eval import distill_run_transductive, distill_run_inductive #t
from data_preprocess import cat_deepwalk_embeddings


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    """环境"""
    parser.add_argument("--device", type=int, default=-1, help="CUDA device, -1 means CPU") #设备选择
    # parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU") #设备选择
    parser.add_argument("--seed", type=int, default=0, help="Random seed") #随机种子
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    ) #日志级别
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    ) #是否在控制台显示日志信息
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    ) # 输出路径
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    ) #重复实验次数
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    ) #实验设置（tran | ind）
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    ) #每多少个epoch评估一次
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )# 是否保存结果

    """
    (选定的)数据集：名(cora)、所在路径(./data)、训练集标签比例(0.2)、验证集标签比例(0.3)、spilt_idx(one of [0,1,2,3,4])
    """
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )

    """
    (选定的)模型：模型配置文件路径(./train.conf.yaml)、teacher模型(SAGE)、student模型(MLP)/
                学生模型层数(2)、隐藏层维度(64)
    """
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./train.conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Student model number of layers"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,  #t
        help="Student model hidden layer dimensions",
    )
    parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
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

    """Distiall 学生模型特有参数""" #t
    parser.add_argument( #用于平衡硬标签的损失和教师输出的损失（0-1之间）
        "--lamb",
        type=float,
        default=0,
        help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]",
    )
    parser.add_argument( #预训练教师模型的输出结果路径
        "--out_t_path", type=str, default="outputs", help="Path to load teacher outputs"
    )
    args = parser.parse_args()
    print(args)
    return args


def run(args, num_exp):
    '''模型运行主函数'''
    set_seed(args.seed) #设置随机种子
    if torch.cuda.is_available() and args.device >= 0: #设置训练设备
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    '''根据传入的命令行参数(模型训练设置)调整输出路径'''
    if args.feature_noise != 0 and num_exp==1:  #如果特征噪声（args.feature_noise）不为0，那么将会对输出路径（args.output_path）进行修改，并假设教师模型也是在相同的噪声特征上进行训练的
        args.output_path = Path.cwd().joinpath(
            args.output_path, "noisy_features", f"noise_{args.feature_noise}"
        )
        # Teacher is assumed to be trained on the same noisy features as well.
        args.out_t_path = args.output_path

    if args.feature_aug_k > 0 and num_exp == 1: #如果特征增强（args.feature_aug_k）大于0，那么将会对输出路径（args.output_path）进行修改，并假设教师模型也是在相同的增强特征上进行训练的
        args.output_path = Path.cwd().joinpath(
            args.output_path, "aug_features", f"aug_hop_{args.feature_aug_k}"
        )
        # NOTE: Teacher may or may not have augmented features, specify args.out_t_path explicitly.
        # args.out_t_path =
        args.student = f"GA{args.feature_aug_k}{args.student}"

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath( #学生模型输出结果路径
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(  #教师模型输出结果路径（软标签）
            args.out_t_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    check_readable(out_t_dir)

    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"out_t_dir: {out_t_dir}")

    """ 加载数据 """
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

    # feats = g.ndata["feat"]
    # args.feat_dim = g.ndata["feat"].shape[1]
    feats = cat_deepwalk_embeddings(g)  #将DeepWalk的节点向量表示拼接到节点特征矩阵中
    args.feat_dim = feats.shape[1]
    args.label_dim = labels.int().max().item() + 1

    '''在处理特征噪声的情况: 若feature_noise∈(0,1]则对特征进行随机噪声处理)
        噪声的级别（args.feature_noise）决定了噪声的强度，值越大，噪声越强，特征的原始信息保留得越少
    '''
    if 0 < args.feature_noise <= 1:
        feats = (
            1 - args.feature_noise
        ) * feats + args.feature_noise * torch.randn_like(feats)

    """ 模型设置 """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(  #获取训练配置conf
            args.model_config_path, args.student, args.dataset
        )  # Note: student config
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")

    """ 模型创建和初始化 """
    model = Model(conf)  #传入配置，创建模型
    # for param in model.parameters():  # 查看模型参数
    #     print("参数", param, type(param), param.size())
    optimizer = optim.Adam(  #创建优化器
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion_l = torch.nn.NLLLoss() #NLLLoss损失函数
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True) #KL散度损失函数
    evaluator = get_evaluator(conf["dataset"]) #根据模型，获取对应的评估器

    """加载teacher模型输出"""
    out_t = load_out_t(out_t_dir)
    logger.debug(
        f"teacher score on train data: {evaluator(out_t[idx_train], labels[idx_train])}"
    )
    logger.debug(
        f"teacher score on val data: {evaluator(out_t[idx_val], labels[idx_val])}"
    )
    logger.debug(
        f"teacher score on test data: {evaluator(out_t[idx_test], labels[idx_test])}"
    )

    """Data split and run"""
    loss_and_score = []
    if args.exp_setting == "tran":  #直推设置
        idx_l = idx_train  #半监督学习中，训练集的数据是带标签的，用于训练模型
        idx_t = torch.cat([idx_train, idx_val, idx_test]) #直推学习的设置下，模型在训练过程中可以看到所有的数据（包括训练集、验证集和测试集），但只有训练集的数据是带标签的
        distill_indices = (idx_l, idx_t, idx_val, idx_test) #这个元组后续可能会被用于指导模型的训练过程，例如，idx_l可能会被用于计算损失函数，idx_t可能会被用于进行特征传播，idx_val和idx_test可能会被用于在验证集和测试集上评估模型的性能
        '''这样设置的目的是为了方便后续的模型训练和评估过程。
        通过预先设置好各个数据集的索引，可以在后续的代码中直接使用这些索引，从而简化代码的复杂度。
        '''

        # propagate node feature
        if args.feature_aug_k > 0:
            feats = feature_prop(feats, g, args.feature_aug_k)

        out, score_val, score_test = distill_run_transductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test]

    elif args.exp_setting == "ind":
        # Create inductive split
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(
            idx_train, idx_val, idx_test, args.split_rate, args.seed
        )
        obs_idx_l = obs_idx_train
        obs_idx_t = torch.cat([obs_idx_train, obs_idx_val, obs_idx_test])
        distill_indices = (
            obs_idx_l,
            obs_idx_t,
            obs_idx_val,
            obs_idx_test,
            idx_obs,
            idx_test_ind,
        )

        # propagate node feature. The propagation for the observed graph only happens within the subgraph obs_g
        if args.feature_aug_k > 0:
            obs_g = g.subgraph(idx_obs)
            obs_feats = feature_prop(feats[idx_obs], obs_g, args.feature_aug_k)
            feats = feature_prop(feats, g, args.feature_aug_k)
            feats[idx_obs] = obs_feats

        out, score_val, score_test_tran, score_test_ind = distill_run_inductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            distill_indices,
            criterion_l,
            criterion_t,
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

    """ Saving student outputs """
    out_np = out.detach().cpu().numpy()
    np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss"""
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
            f.write(f"{min_cut :.4f}\n")

    return score_lst


def repeat_run(args):  #多次运行训练的实验
    scores = []
    for seed in range(args.num_exp):
        args.seed = seed
        scores.append(run(args, seed+1))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)  #返回所有分数结果(预测精度)的平均值和标准差


def main():
    args = get_args()
    if args.num_exp == 1:
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
