import numpy as np
import torch
import logging
import pytz  #python自带的库，用于处理时区
import random
import os
import yaml
import shutil
from datetime import datetime
from ogb.nodeproppred import Evaluator
from dgl import function as fn

CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]
NonHom_data = ["pokec", "penn94"]
BGNN_data = ["house_class", "vk_class"]


def set_seed(seed):
    '''
    设置随机种子，确保实验的可重复性
    '''
    torch.manual_seed(seed)  #CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  #GPU
    np.random.seed(seed)
    random.seed(seed)  #Numpy和Python内置随机数生成器
    '''
    设置了 PyTorch 的 cuDNN 后端的两个选项：benchmark 和 deterministic——
    不启用 cuDNN 的自动优化——cuDNN 的自动优化会在程序开始运行时，自动寻找最适合当前配置（如输入数据的大小和卷积层的参数）的高效算法。这通常可以提高运行速度，但是在某些情况下，可能会导致结果的不一致性；
    启用 cuDNN 的确定性模式——确定性模式下，cuDNN 会使用确定性的算法，确保每次运行程序时，对于相同的输入，都会得到完全相同的结果。这对于确保实验的可重复性非常重要
    '''
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True  # 禁用CuDNN的性能测试，以确保结果的确定性

def get_training_config(config_path, model_name, dataset):
    '''
    参数：config_path-配置文件路径；model_name-模型名称；dataset-数据集名称
    返回：specific_config-（本次特定的）训练配置
    '''
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)  # 读取配置文件
    dataset_specific_config = full_config["global"]  # 全局配置
    model_specific_config = full_config[dataset][model_name] # 具体模型配置

    if model_specific_config is not None: 
        specific_config = dict(dataset_specific_config, **model_specific_config)
    else:
        specific_config = dataset_specific_config
    # specific_config = model_specific_config

    specific_config["model_name"] = model_name
    print('训练配置：\n' + str(specific_config))
    return specific_config


def check_writable(path, overwrite=True):
    if not os.path.exists(path):  #当前目录下路径不存在，就新建
        os.makedirs(path)
    elif overwrite:  # 存在且允许重写，则删除重建
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def check_readable(path):
    if not os.path.exists(path):  #当前目录下路径不存在，就报错
        raise ValueError(f"No such file or directory! {path}")


def timetz(*args):
    '''
    获取美国太平洋时区的当前时间，然后以时间元组的形式返回。
    时间元组是一个包含 9 个元素的元组：年、月、日、小时、分钟、秒、一周中的第几天、一年中的第几天和夏令时标志
    '''
    tz = pytz.timezone("US/Pacific") #获取中国时间：pytz.timezone("Asia/Shanghai")
    return datetime.now(tz).timetuple()


def get_logger(filename, console_log=True, log_level=logging.INFO):
    '''
    传入日志配置，返回logger实例
    参数：filename-日志文件名；console_log-是否在控制台输出日志；log_level-日志级别
    返回：logger-日志记录器
    '''
    # tz = pytz.timezone("US/Pacific")
    # '''用于处理世界时区（接受一个时区名称作为参数，并返回一个代表该时区的 datetime.tzinfo 对象）
    #     1.在中国可以使用："Asia/Shanghai"
    # '''
    # log_time = datetime.now(tz).strftime("%b%d_%H_%M_%S") #获取当前时间，并以指定格式返回
    # 这段在原程序中并没有使用，logger的时间格式是在formatter中设置的，formatter.converter在后面被设置为自定义函数timetz

    logger = logging.getLogger(__name__) #创建一个logger实例（日志记录器，以当前模块名）
    logger.propagate = False  # 设置日志传播标志，False，则日志事件只在当前的 logger 上进行处理，不会传递给父 logger。避免日志信息被重复记录
    logger.setLevel(log_level) #设置日志记录器的日志级别(logging.DEBUG、logging.INFO、logging.WARNING、logging.ERROR、logging.CRITICAL)——决定了哪些级别的日志信息会被记录：如果日志级别被设置为 logging.INFO，那么 info、warning、error 和 critical 级别的日志信息会被记录，而 debug 级别的日志信息则会被忽略

    for hdlr in logger.handlers[:]:  #配置日志处理器之前移除已有处理器（防止旧处理器影响新配置）
        logger.removeHandler(hdlr)
    
    file_handler = logging.FileHandler(filename)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
    '''日志格式器：指定日志格式
        1."%(asctime)s: %(message)s": 指定日志中的每行内容——%(asctime)s 和 %(message)s 是两个格式化字段。%(asctime)s 会被日志记录的时间替换，%(message)s 会被日志记录的消息文本替换
        2.datefmt="%b%d %H-%M-%S": 指定日志记录的时间格式——"%b%d %H-%M-%S" 是一个格式化字符串，用于指定日志记录的时间格式。其中，%b 表示月份的简写，%d 表示日期，%H 表示小时，%M 表示分钟，%S 表示秒
    '''
    formatter.converter = timetz #格式器的时间转换函数
    file_handler.setFormatter(formatter)  # 将格式器交给日志处理器
    logger.addHandler(file_handler)   # 将日志处理器添加到日志记录器(logger示例)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def idx_split(idx, ratio, seed=0):
    """
    传入索引列表和划分比例，根据比例将索引列表划分为两部分 (该方法专为graph_split调用)
    参数：idx-索引列表；ratio-划分索引比例；seed-随机种子
    返回：划分后的两部分索引列表idx1, idx2
    """
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    # assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2


def graph_split(idx_train, idx_val, idx_test, rate, seed):
    """
    Args:
        The original setting was transductive. Full graph is observed, and idx_train takes up a small portion.
        Split the graph by further divide idx_test into [idx_test_tran, idx_test_ind].
        rate = idx_test_ind : idx_test (how much test to hide for the inductive evaluation)

        Ex. Ogbn-products
        loaded     : train : val : test = 8 : 2 : 90, rate = 0.2
        after split: train : val : test_tran : test_ind = 8 : 2 : 72 : 18

    Return:
        Indices start with 'obs_' correspond to the node indices within the observed subgraph,
        where as indices start directly with 'idx_' correspond to the node indices in the original graph
    将数据集分割成训练集、验证集、tran测试集、ind测试集，并生成对应的索引
    参数：idx_train-训练集索引；idx_val-验证集索引；idx_test-测试集索引；rate-划分测试集比例；seed-随机种子；
    返回：obs_idx_train-观察到的训练集索引；obs_idx_val-观察到的验证集索引；obs_idx_test-观察到的测试集索引；
    """
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran]) #将训练集索引 idx_train、验证集索引 idx_val 和测试集索引 idx_test_tran 拼接成一个新的索引列表 idx_obs
    N1, N2 = idx_train.shape[0], idx_val.shape[0] #训练集索引和验证集索引的长度
    obs_idx_all = torch.arange(idx_obs.shape[0]) 
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1 : N1 + N2]
    obs_idx_test = obs_idx_all[N1 + N2 :]

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def get_evaluator(dataset):
    '''根据数据集返回对应的评估器'''
    if dataset in CPF_data + NonHom_data + BGNN_data:
        def evaluator(out, labels):
            '''
            根据模型输出和真实标签计算预测准确度
            参数：out-模型输出；labels-真实标签
            返回：预测准确度
            '''
            pred = out.argmax(1) #计算 out 的每一行的最大值的索引，也就是找出每个样本得分最高的类别。这是多分类问题中常用的一种方法
            return pred.eq(labels).float().mean().item() #pred.eq(labels) 比较预测结果和真实标签是否相等，得到一个布尔型张量。然后，float() 将布尔型张量转换为浮点型张量，True 转换为 1.0，False 转换为 0.0。接着，mean() 计算张量的平均值，得到预测准确率。最后，item() 将张量转换为 Python 的标量

    elif dataset in OGB_data:
        ogb_evaluator = Evaluator(dataset) #获取OGB数据集的评估器

        def evaluator(out, labels):
            pred = out.argmax(1, keepdim=True)
            input_dict = {"y_true": labels.unsqueeze(1), "y_pred": pred}
            return ogb_evaluator.eval(input_dict)["acc"]

    else:
        raise ValueError("Unknown dataset")

    return evaluator


# def get_evaluator(dataset):
#     def evaluator(out, labels):
#         pred = out.argmax(1)
#         return pred.eq(labels).float().mean().item()

#     return evaluator

# def get_parameter_number(model):
#     '''计算模型的参数数量'''
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}

def compute_min_cut_loss(g, out):
    '''
    参数：g-图(DGLGraph)；out-模型输出（2维tensor）
    返回：最小割损失（标量）
    【注】最小割损失：
        1.一种评估图划分质量的指标
        2.图划分问题中，我们的目标是将图的节点划分为多个组，使得组内的边尽可能多，组间的边尽可能少；
          最小割损失就是用来度量组间边的数量
        3.最小割损失是指所有划分边的权重之和。
          最小割损失越小，说明图的划分越好。
    '''
    out = out.to("cpu")  #将out(模型输出，二维张量)移动到cpu上
    S = out.exp()  # 对out进行指数运算
    A = g.adj().to_dense() #将图的邻接矩阵转换为密集张量
    D = g.in_degrees().float().diag() #将图的入度转换为浮点型张量，并生成对角矩阵
    min_cut = (
        torch.matmul(torch.matmul(S.transpose(1, 0), A), S).trace()
        / torch.matmul(torch.matmul(S.transpose(1, 0), D), S).trace()
    )
    return min_cut.item()


def feature_prop(feats, g, k):
    """
    在 k跳 邻域内传播节点特征来更新节点特征，这种传播是以 SGC（Simplified Graph Convolution）的方式进行的，即逐跳（hop by hop）进行，并通过节点度对其进行对称归一化。
    1.节点特征的传播是以 SGC 的方式进行的：SGC 是一种图卷积网络，它简化了图卷积的计算过程，使得特征传播可以逐跳进行，而不需要在每一跳都进行复杂的卷积计算
    2.每一跳的传播过程中，都会对节点特征进行对称归一化（这种归一化是通过节点的度（即与节点相连的边的数量）来进行的，可以防止特征值的爆炸或消失，从而使得特征传播更稳定）
    参数：feats-节点特征矩阵；g-图；k-传播次数
    返回：增强后的节点特征矩阵
    【注】SGC和传统图卷积的主要区别：
            1.SGC 去掉了激活函数，使得特征传播可以直接进行，无需在每一跳都进行非线性变换
            2.在传统的图卷积网络中，每一层都有一个权重矩阵，用于在特征传播过程中对特征进行线性变换。这个权重矩阵是模型的参数，需要通过训练数据来学习。然而，SGC 去掉了这个权重矩阵，使得特征传播可以直接进行，无需在每一跳都进行线性变换
    """
    assert feats.shape[0] == g.num_nodes()  #确保特征矩阵行数与图的节点数相等

    degs = g.in_degrees().float().clamp(min=1) #计算图的入度，并将入度小于1的节点的入度设为1（避免后续计算中的除以零错误）
    norm = torch.pow(degs, -0.5).unsqueeze(1) #计算节点度数的负平方根，并增加一个维度（将结果扩展到二维张量，以便进行矩阵运算）

    # compute (D^-1/2 A D^-1/2)^k X
    for _ in range(k):  #进行 k 次特征传播
        feats = feats * norm
        '''使得每个节点的特征在传播过程中被其度数的平方根的倒数所缩放,
           目的是为了防止节点特征在传播过程中的放大或缩小,
           通过norm这个归一化因子，我们可以确保每个节点的特征在传播过程中保持在一个合理的范围内。
        '''
        g.ndata["h"] = feats #将特征矩阵赋值给图的节点特征
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        '''将每个节点(设为节点v)的 "h" 特征传播到其所有邻居节点，并将所有邻居节点的 "h" 特征求和，然后将结果保存在节点v的 "h" 特征中：
            1.fn.copy_u("h", "m")：消息函数，将源节点（"u"）的 "h" 特征复制到消息（"m"）中。这里 "h" 是节点的特征，"m" 是消息。
            2.fn.sum("m", "h")：聚合函数，表示将所有进入目标节点（"v"）的消息（"m"）求和，并将结果保存在目标节点的 "h" 特征中。
            3.update_all(消息函数, 聚合函数, 更新函数(optional))：消息传递高级接口
        '''
        feats = g.ndata.pop("h") #将传播后的特征矩阵赋值给feats
        feats = feats * norm  #将特征矩阵与节点度数的负平方根相乘

    return feats #返回增强后的节点特征矩阵
