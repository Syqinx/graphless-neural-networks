"""
本脚本用于数据预处理，包括：
    1. 读取数据集
    2. 构建邻接矩阵、特征矩阵、标签矩阵
    3. 构建训练集、验证集、测试集索引及其邻接矩阵、特征矩阵、标签矩阵

Adapted from the CPF implementation  改编自CPF实现（附上代码链接）
https://github.com/BUPT-GAMMA/CPF/tree/389c01aaf238689ee7b1e5aba127842341e123b6/data
注释中的 URL 是 CPF 实现的 GitHub 仓库地址，特定的提交哈希（389c01aaf238689ee7b1e5aba127842341e123b6）指向了代码改编的原始版本。
这种注释在开源项目中很常见，用于给出代码来源的引用，以便其他开发者可以查看原始代码，了解更多的上下文信息，或者找到可能的更新和改进
"""

import numpy as np
import scipy.sparse as sp
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dgl.sampling import random_walk
from gensim.models import Word2Vec

def cat_deepwalk_embeddings(g):
    """使用deepwallk进行图嵌入，将学习到的向量表示和原特征矩阵进行拼接，以让模型输入考虑图结构信息而非仅仅是节点特征
    
    参数：g-原始图数据
    返回：feats-拼接后的节点特征矩阵
    """
    # 2. 随机游走生成节点序列
    walks,_ = random_walk(g, g.nodes(), length=10)
    walks = walks.tolist()
    walks = [[str(node) for node in walk] for walk in walks]
    # 3. 使用 Word2Vec 学习节点向量表示
    model = Word2Vec(walks, size=64, window=5, min_count=0, sg=1, workers=4)
    # 获取节点向量表示
    embeddings = [model.wv[str(i)] for i in range(g.number_of_nodes())]
    embeddings = torch.tensor(embeddings)
    feats = torch.cat((torch.tensor(g.ndata['feat']), embeddings), dim=1)
    print('特征拼接完毕！')
    return feats
#
def is_binary_bag_of_words(features):
    '''检查节点特征矩阵是否是二元词袋模型(未调用)'''
    features_coo = features.tocoo()
    return all(
        single_entry == 1.0
        for _, _, single_entry in zip(
            features_coo.row, features_coo.col, features_coo.data
        )
    )
    '''
    1.`for _, _, single_entry in zip(……)` 遍历 features_coo 矩阵的所有非零元素
    2.`single_entry == 1.0` 检查每个非零元素的值是否等于 1.0
    3. all(……) 如果所有非零元素的值都等于 1.0，则返回 True，否则返回 False
    '''

#
def to_binary_bag_of_words(features):
    """将 TF/IDF 特征转换为二元词袋（Binary Bag-of-Words）特征（未调用）
        1.TF/IDF（Term Frequency-Inverse Document Frequency）是一种常用的文本特征提取方法——考虑了每个词在文档中的频率（TF）和在整个文档集中的逆文档频率（IDF）
        2.二元词袋模型是一种文本表示方法，其中每个文档表示为一个向量，向量的每个元素对应一个词汇，如果该词汇在文档中出现，则对应的元素值为 1，否则为 0
    """
    features_copy = features.tocsr() #将特征矩阵features转换为csr格式的稀疏矩阵
    features_copy.data[:] = 1.0 #features_copy 矩阵的所有非零元素的值设置为 1.0（因为邻接矩阵只存储非零元素）
    return features_copy


def normalize(mx):
    """对稀疏矩阵进行行归一化（由本文件normalize_adj调用）"""
    rowsum = np.array(mx.sum(1)) #计算了矩阵 mx 的每一行的和
    r_inv = np.power(rowsum, -1).flatten() #每一行和的倒数，并将结果转换为一个一维数组
    r_inv[np.isinf(r_inv)] = 0.0 #将 r_inv 中的无穷大值（inf）替换为 0.0
    r_mat_inv = sp.diags(r_inv) #生成一个对角矩阵，对角线上的元素为 r_inv
    mx = r_mat_inv.dot(mx) #对稀疏矩阵 mx 进行行归一化
    return mx


def normalize_adj(adj):
    '''对邻接矩阵添加自环后进行行归一化（由dataloader.py调用，用于对邻接矩阵进行预处理——添加自环并行归一化）'''
    adj = normalize(adj + sp.eye(adj.shape[0])) #`adj+sp.eye(adj.shape[0])`给图添加自环
    return adj


def eliminate_self_loops_adj(A):
    """消除邻接矩阵中的自环（由dataloader.py调用，用于消除自环——对邻接矩阵进行预处理）
    参数：A-邻接矩阵(CSR格式，稀疏矩阵)
    """
    A = A.tolil() #将稀疏矩阵转格式 CSR ->  LIL（List of Lists）
    A.setdiag(0) #将对角线元素设置为0
    A = A.tocsr() #将稀疏矩阵转回 CSR（Compressed Sparse Row）格式
    A.eliminate_zeros() # 删除零元素（稀疏矩阵不存储零元素）
    return A


def largest_connected_components(sparse_graph, n_components=1):
    """获取图的最大连通分量（由dataloader.py调用，用于处理图数据集<确保图中节点都是连通的>）
    参数：sparse_graph-稀疏图；n_components-选择连通分量的个数（默认为1，代表只返回最大连通分量）
    返回：图的最大连通分量（仍是一个稀疏图）
    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix) #component_indices 是一个数组，其中每个元素表示对应节点所属的连通分量的索引
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][
        :n_components
    ]  # reverse order to sort descending
    nodes_to_keep = [
        idx
        for (idx, component) in enumerate(component_indices)
        if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(
    sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None
):
    """使用指定节点子集创建子图（由本文件的largest_connected_components和remove_underrepresented_classes函数调用）
    参数 nodes_to_remove, nodes_to_keep之中其一给定，另一留空(不能同时给定)
    _sentinel：一个内部参数，用于防止传递位置参数。你不应该使用这个参数。
    返回：指定节点被删除的子图
    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError(
            "Only call `create_subgraph` with named arguments',"
            " (nodes_to_remove=...) or (nodes_to_keep=...)"
        )
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError(
            "Only one of nodes_to_remove or nodes_to_keep must be provided."
        )
    elif nodes_to_remove is not None:
        nodes_to_keep = [
            i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove
        ]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """将标签矩阵转换为二值矩阵（每行代表一个样本、每列表示一个类别<属于为1>）（由dataloader.py调用，用来预处理节点标签矩阵）
    labels : 输入的标签，可以是单标签(数组)或多标签格式(矩阵)
    sparse_output : 是否返回CSR格式的标签矩阵
    return_classes : 是否返回类别数组

    label_matrix : 二进制的标签矩阵，每一行对应一个样本，每一列对应一个类别（如果样本属于某个类别，那么对应的元素值为 1，否则为 0）   
    classes : 返回类别数组

    """
    if hasattr(labels[0], "__iter__"):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    '''创建标签二值化器：（sklearn的评估器<用于数据预处理>）
    先检查 labels[0] 是否是可迭代的：
        若可迭代，说明labels 是多标签格式（因为Python 中，字符串、列表、元组等都是可迭代的）
        若不可迭代，说明labels 是单标签格式
    '''
    label_matrix = binarizer.fit_transform(labels).astype(np.float32) #进行处理转换
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix

#
def remove_underrepresented_classes(
    g, train_examples_per_class, val_examples_per_class
):
    """
    移除图中对应的类别样本数量少于 num_classes * train_examples_per_class + num_classes * val_examples_per_class 的节点（如果某个类别的节点数量过少，那么在训练和验证过程中可能无法得到足够的样本，这会影响模型的训练效果）

    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(
        class_
        for class_, count in examples_counter.items()
        if count > min_examples_per_class
    )
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)
