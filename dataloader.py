"""
Dataloader of CPF datasets are adapted from the CPF implementation
https://github.com/BUPT-GAMMA/CPF/tree/389c01aaf238689ee7b1e5aba127842341e123b6/data

Dataloader of NonHom datasets are adapted from the Non-homophily benchmarks
https://github.com/CUAI/Non-Homophily-Benchmarks

Dataloader of BGNN datasets are adapted from the BGNN implementation and dgl example of BGNN
https://github.com/nd7141/bgnn
https://github.com/dmlc/dgl/tree/473d5e0a4c4e4735f1c9dc9d783e0374328cca9a/examples/pytorch/bgnn
"""

import os
import json
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp  # 稀疏矩阵模块
import torch
import dgl
from dgl.data.utils import load_graphs
from os import path
from category_encoders import CatBoostEncoder
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from data_preprocess import (
    normalize_adj,
    eliminate_self_loops_adj,
    largest_connected_components,
    binarize_labels,
)
from ogb.nodeproppred import DglNodePropPredDataset

CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"] # 运行dataloader.py 中的load_data函数时，会自动下载ogbn-arxiv和ogbn-products数据集
# 运行dataloader.py 中的load_data函数时，会自动下载pokec数据集（分析源码可知：是从https://docs.google.com/uc?export=download上下载，因此需要科学上网）
NonHom_data = ["pokec", "penn94"]  
BGNN_data = ["house_class", "vk_class"] 


def load_data(dataset, dataset_path, **kwargs):
    '''
    参数：dataset-数据集名称(str)；dataset_path-数据集路径(str)；**kwargs-可变关键字参数
    返回：g-DGLGraph对象, labels-标签矩阵, idx_train-训练集的索引, idx_val-验证集的索引, idx_test0测试集的索引
    '''
    #加载数据集
    if dataset in CPF_data: # 如果是CPF中的数据集
        return load_cpf_data(  #见后
            dataset,
            dataset_path,
            kwargs["seed"],
            kwargs["labelrate_train"],
            kwargs["labelrate_val"],
        )
    elif dataset in OGB_data: # 如果是OGB中的数据集
        return load_ogb_data(dataset, dataset_path) #见后
    elif dataset in NonHom_data: # 如果是NonHom中的数据集
        return load_nonhom_data(dataset, dataset_path, kwargs["split_idx"]) #见后
    elif dataset in BGNN_data: # 如果是BGNN中的数据集
        return load_bgnn_data(dataset, dataset_path, kwargs["split_idx"]) #见后
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def load_cpf_data(dataset, dataset_path, seed, labelrate_train, labelrate_val):
    """
    加载CPF数据集的底层函数
    参数：dataset 数据集名称(str)；dataset_path 数据集路径(str)；seed 随机种子(int)；labelrate_train 训练集标签比例([0.0, 1.0)的浮点数)；labelrate_val 验证集标签比例
    返回值：g-DGLGraph, labels-节点标签矩阵, idx_train、idx_val、idx_test-训练、验证、测试集索引
    """
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.npz")  # 拼接数据集本地路径
    if os.path.isfile(data_path):
        data = load_npz_to_sparse_graph(data_path)  #load_npz_to_sparse_graph(见后)加载npz数据变成稀疏图sparse
    else:
        raise ValueError(f"{data_path} doesn't exist.")

    data = data.standardize()  #图的标准化: 无向化、无权化、去自环，最终返回该图的最大连通子图
    adj, features, labels = data.unpack() # 将CPF(稀疏)图拆分为邻接矩阵、属性矩阵、标签矩阵返回

    labels = binarize_labels(labels) #将输入的labels矩阵二值化（一般来说，二值化是将所有正类标签转换为 1，所有负类标签转换为 0，具体作用见实现）

    random_state = np.random.RandomState(seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(
        random_state, labels, labelrate_train, labelrate_val
    ) #
    #格式转换
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels.argmax(axis=1))

    adj = normalize_adj(adj) #邻接矩阵adj添加自环后行归一化
    adj_sp = adj.tocoo() #将邻接矩阵adj转换为COO的稀疏矩阵格式
    g = dgl.graph((adj_sp.row, adj_sp.col))  #输入邻接矩阵的行和列，返回DGLGraph对象
    g.ndata["feat"] = features  #图的节点特征

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return g, labels, idx_train, idx_val, idx_test

def load_ogb_data(dataset, dataset_path):
    """
    加载ogb库的数据集的底层函数：
    参数：dataset 数据集名(str)；dataset_path 数据集路径(str)
    返回值：g-, labels, idx_train, idx_val, idx_test
    """
    data = DglNodePropPredDataset(dataset, dataset_path) #加载一个用于节点属性预测任务的数据集
    splitted_idx = data.get_idx_split()  #返回一个3个元素的字典：训练集、验证集、测试集
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    g, labels = data[0]
    labels = labels.squeeze()  #去除大小为1的维度

    # 将图形转为无向图
    if dataset == "ogbn-arxiv":
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()

    return g, labels, idx_train, idx_val, idx_test

def load_nonhom_data(dataset, dataset_path, split_idx):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.mat")
    data_split_path = Path.cwd().joinpath(
        dataset_path, "splits", f"{dataset}-splits.npy"
    )

    if dataset == "pokec":
        g, features, labels = load_pokec_mat(data_path)
    elif dataset == "penn94":
        g, features, labels = load_penn94_mat(data_path)
    else:
        raise ValueError("Invalid dataname")

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = torch.LongTensor(labels)

    splitted_idx = load_fixed_splits(dataset, data_split_path, split_idx)
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    return g, labels, idx_train, idx_val, idx_test

def load_bgnn_data(dataset, dataset_path, split_idx):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}")

    g, X, y, cat_features, masks = read_input(data_path)
    train_mask, val_mask, test_mask = (
        masks[str(split_idx)]["train"],
        masks[str(split_idx)]["val"],
        masks[str(split_idx)]["test"],
    )

    encoded_X = X.copy()
    if cat_features is not None and len(cat_features):
        encoded_X = encode_cat_features(
            encoded_X, y, cat_features, train_mask, val_mask, test_mask
        )
    encoded_X = normalize_features(encoded_X, train_mask, val_mask, test_mask)
    encoded_X = replace_na(encoded_X, train_mask)
    features, labels = pandas_to_torch(encoded_X, y)

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = labels.long()

    idx_train = torch.LongTensor(train_mask)
    idx_val = torch.LongTensor(val_mask)
    idx_test = torch.LongTensor(test_mask)
    return g, labels, idx_train, idx_val, idx_test

def load_out_t(out_t_dir):  
    '''
    加载out_t_dir(教师模型输出)目录下的out.npz文件
    '''
    # t_data = np.load(out_t_dir.joinpath("out.npz")) #返回NpzFile对象
    # print(t_data.files) #打印出 t_data 中的所有文件名（.npz 格式的文件中，可以包含多个数组，每个数组都有一个名字）（返回结果：['arr_0']）
    return torch.from_numpy(np.load(out_t_dir.joinpath("out.npz"))["arr_0"])


""" NonHom的数据集：penn94、pokec"""
dataset_drive_url = {"pokec": "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y"}
splits_drive_url = {"pokec": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_"}
def load_penn94_mat(data_path):
    """
    加载penn94数据集的底层函数：
    参数：data_path 数据集路径(str)
    返回值：g, features, labels
    """
    mat = scipy.io.loadmat(data_path)
    # print(type(mat)) # <class 'dict'>

    # print(scipy.io.whosmat(data_path))  # 返回指定路径的.mat文件中的变量信息(list)  [('A', (41554, 41554), 'sparse'), ('local_info', (41554, 7), 'double')]
    # print(mat.keys())
    """
    dict_keys(['__header__', '__version__', '__globals__', 'A', 'local_info']) 
    - mat['__header__']：b'MATLAB 5.0 MAT-file, Platform: MACI64, Created on: Fri Feb 11 20:20:46 2011'
    - mat['__version__']：1.0
    - mat['__globals__']：[]
    - mat['local_info']：
    """
    A = mat["A"]
    # print(type(A))
    metadata = mat["local_info"]
    # print(type(metadata)) # <class 'numpy.ndarray'>

    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)

    # make features into one-hot encodings
    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(metadata[:, 1] - 1)  # gender label, -1 means unlabeled
    return g, features, labels

def load_pokec_mat(data_path):
    if not path.exists(data_path):
        gdd.download_file_from_google_drive( #从谷歌云盘(google driver)下载文件
            file_id=dataset_drive_url["pokec"], dest_path=data_path, showsize=True
        )

    fulldata = scipy.io.loadmat(data_path)
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(fulldata["node_feat"]).float()
    labels = fulldata["label"].flatten()
    return g, features, labels

""" For CPF"""
class SparseGraph:
    """属性、标签图（稀疏矩阵形式）"""
    def __init__(
        
        self,
        adj_matrix, #csr格式的邻接矩阵
        attr_matrix=None, #属性矩阵
        labels=None, #
        node_names=None, #
        attr_names=None, #
        class_names=None,
        metadata=None,
    ):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format. 邻接矩阵（CSR格式）
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format. 属性矩阵(csr格式或numpy格式)
        labels : np.ndarray, shape [num_nodes], optional 
            Array, where each entry represents respective node's label(s).  每个节点的标签
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).  节点的名称（str）
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).  属性的名称（str）
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).  #类别标签的名称（str）
        metadata : object
            Additional metadata such as text.  附加元数据，如文本

        """
        #保证邻接矩阵adj_matrix是稀疏格式（因为实际世界中大多数图都是稀疏的）
        if sp.isspmatrix(adj_matrix): 
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError(
                "Adjacency matrix must be in sparse format (got {0} instead)".format(
                    type(adj_matrix)
                )
            )
        #保证adj_matrix是方阵（邻接矩阵的强制要求）
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError(
                    "Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(
                        type(attr_matrix)
                    )
                )

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency and attribute matrices don't agree"
                )

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the label vector don't agree"
                )

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the node names don't agree"
                )

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError(
                    "Dimensions of the attribute matrix and the attribute names don't agree"
                )

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """图中节点的数量"""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """边的数量.
        （对于无向图, (i, j)和(j, i)算作一条边）
        """
        if self.is_directed():  
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        '''
        输入：idx-某节点索引
        返回；该节点的邻居节点索引
        【注】CSR 格式的稀疏矩阵有三个主要的属性：data、indices 和 indptr（均为1维numpy数组）
            1.data：矩阵中所有非零元素的值
            2.indices：矩阵中所有非零元素的列索引
            3.indptr：行的第一个非零元素在 data 和 indices 中的索引
        '''
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """是否是有向图"""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """转换为无向图（首先转换为无权图了）"""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1  #`self.adj_matrix!=0`返回与adj_matrix同形状的bool矩阵
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """转换为无权图"""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.
        选择无权、无向、无自环图的最大连通分量
        All changes are done inplace.原地修改
        """
        G = self.to_unweighted().to_undirected()  #将图 -》无权图 -》无向图
        G.adj_matrix = eliminate_self_loops_adj(G.adj_matrix) #删除自环
        G = largest_connected_components(G, 1) #选择最大连通分量
        return G

    def unpack(self):
        """Return the (A, X, z) triplet.
        将CPF(稀疏)图拆分为邻接矩阵、属性矩阵、标签矩阵返回
        """
        return self.adj_matrix, self.attr_matrix, self.labels

def load_npz_to_sparse_graph(file_name):
    """
    从numpy二进制文件(.npz)中加载稀疏图(矩阵)。
    参数：file_name-文件名路径(str)
    返回：
    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        if "attr_data" in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix(
                (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
                shape=loader["attr_shape"],
            )
        elif "attr_matrix" in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader["attr_matrix"]
        else:
            attr_matrix = None

        if "labels_data" in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix(
                (
                    loader["labels_data"],
                    loader["labels_indices"],
                    loader["labels_indptr"],
                ),
                shape=loader["labels_shape"],
            )
        elif "labels" in loader:
            # Labels are stored as a numpy array
            labels = loader["labels"]
        else:
            labels = None

        node_names = loader.get("node_names")
        attr_names = loader.get("attr_names")
        class_names = loader.get("class_names")
        metadata = loader.get("metadata")

    return SparseGraph(
        adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata
    )

def sample_per_class(
    random_state, labels, num_examples_per_class, forbidden_indices=None
):
    """获取每个类固定数量的样本（由本文件get_train_val_test_split函数调用）
    参数：
    返回：
    """

    num_samples, num_classes = labels.shape  #标签矩阵的行数(样本数)、列数（分类数）
    sample_indices_per_class = {index: [] for index in range(num_classes)} #创建了一个字典sample_indices_per_class，键是0到num_classes-1，值是空列表
    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [
            random_state.choice(
                sample_indices_per_class[class_index],
                num_examples_per_class,
                replace=False,
            )
            for class_index in range(len(sample_indices_per_class))
        ]
    )

def get_train_val_test_split(
    random_state,
    labels,
    train_examples_per_class=None,
    val_examples_per_class=None,
    test_examples_per_class=None,
    train_size=None,
    val_size=None,
    test_size=None,
):
    '''根据指定的条件，从标签矩阵中选择训练、验证和测试样本的索引（由本文件的load_cpf_data函数调用）
    参数：random_state-随机数种子；labels-节点标签矩阵；train_examples_per_class-,val_examples_per_class=None,test_examples_per_class=None,train_size=None,val_size=None,test_size=None,
    返回：
    '''
    num_samples, num_classes = labels.shape  #获取标签矩阵的行数(样本数)、列数（类别数量）
    remaining_indices = list(range(num_samples)) #包含所有样本索引的列表
    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False
        )

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state,
            labels,
            val_examples_per_class,
            forbidden_indices=train_indices,
        )
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(
            random_state,
            labels,
            test_examples_per_class,
            forbidden_indices=forbidden_indices,
        )
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    #确保train、val、test之间不重不漏
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert (
            len(np.concatenate((train_indices, val_indices, test_indices)))
            == num_samples
        )

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

#
class NCDataset(object):
    '''
    (该类并未使用)
    是一个数据加载器，用于处理和加载图形数据。
    它基于Open Graph Benchmark（OGB）的NodePropPredDataset数据集，但返回的是PyTorch张量而不是NumPy数组，使其更适合用于PyTorch模型的训练。

    '''
    def __init__(self, name, root):
        """
        based off of ogb NodePropPredDataset 基于ogb的NodePropPredDataset数据集
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset  数据集名称
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,but when something is passed, it uses its information. Useful for debugging for external contributers.
        构造函数：
            参数：name-数据集名称；root-本地存储数据集根目录；meta_dict-存储有关数据的所有元信息的字典（默认为None，但当传入时，它使用其信息。对于调试和外部贡献者很有用）。
            返回：

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def rand_train_test_idx(label, train_prop, valid_prop, ignore_negative):
        # """
        # Randomly splits the dataset into train, validation, and test sets.
        # """

        # if ignore_negative:
        #     label = label[label >= 0]

        # num_nodes = len(label)
        # num_train = int(train_prop * num_nodes)
        # num_valid = int(valid_prop * num_nodes)
        # num_test = num_nodes - num_train - num_valid

        # idx = np.random.permutation(num_nodes)

        # train_idx = idx[:num_train]
        # valid_idx = idx[num_train : num_train + num_valid]
        # test_idx = idx[num_train + num_valid :]

        # return train_idx, valid_idx, test_idx
        """
            Randomly splits the dataset into train, validation, and test sets.
        """

        if ignore_negative:
            valid_idx = np.where(label >= 0)[0]
        else:
            valid_idx = np.arange(len(label))

        num_nodes = len(valid_idx)
        num_train = int(train_prop * num_nodes)
        num_valid = int(valid_prop * num_nodes)
        num_test = num_nodes - num_train - num_valid

        idx = np.random.permutation(valid_idx)

        train_idx = idx[:num_train]
        valid_idx = idx[num_train : num_train + num_valid]
        test_idx = idx[num_train + num_valid :]

        return train_idx, valid_idx, test_idx

    # 只定义，没有调用（其中的未定义函数不管他）
    def get_idx_split(self, split_type="random", train_prop=0.5, valid_prop=0.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == "random":
            ignore_negative = False if self.name == "ogbn-proteins" else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,
            )
            split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))

def load_fixed_splits(dataset, data_split_path="", split_idx=0):
    '''（给load_nonhom_data()调用）
    参数：dataset, data_split_path="", split_idx=0
    返回：splits
    '''
    if not os.path.exists(data_split_path): 
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset], dest_path=data_split_path, showsize=True
        )

    splits_lst = np.load(data_split_path, allow_pickle=True)
    splits = splits_lst[split_idx]

    for key in splits:
        if not torch.is_tensor(splits[key]):
            splits[key] = torch.as_tensor(splits[key])

    return splits


"""For BGNN """
def pandas_to_torch(*args):
    return [torch.from_numpy(arg.to_numpy(copy=True)).float().squeeze() for arg in args]

def read_input(input_folder):
    X = pd.read_csv(f"{input_folder}/X.csv")
    y = pd.read_csv(f"{input_folder}/y.csv")

    categorical_columns = []
    if os.path.exists(f"{input_folder}/cat_features.txt"):
        with open(f"{input_folder}/cat_features.txt") as f:
            for line in f:
                if line.strip():
                    categorical_columns.append(line.strip())

    cat_features = None
    if categorical_columns:
        columns = X.columns
        cat_features = np.where(columns.isin(categorical_columns))[0]

        for col in list(columns[cat_features]):
            X[col] = X[col].astype(str)

    gs, _ = load_graphs(f"{input_folder}/graph.dgl")
    graph = gs[0]

    with open(f"{input_folder}/masks.json") as f:
        masks = json.load(f)

    return graph, X, y, cat_features, masks

def normalize_features(X, train_mask, val_mask, test_mask):
    min_max_scaler = preprocessing.MinMaxScaler()
    A = X.to_numpy(copy=True)
    A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
    A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
    return pd.DataFrame(A, columns=X.columns).astype(float)

def replace_na(X, train_mask):
    '''处理缺失值'''
    if X.isna().any().any():
        return X.fillna(X.iloc[train_mask].min() - 1)
    return X

def encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask):
    '''对分类特征进行编码
    参数：X-(Pandas DataFrame)特征矩阵；y-(Pandas Series)标签向量；cat_features-分类特征；train_mask-训练集掩码；val_mask-验证集掩码；test_mask-测试集掩码
    返回：pd.DataFrame(A, columns=X.columns) - 编码后的特征矩阵
    '''
    enc = CatBoostEncoder() #创建CatBoost编码器（基于目标变量的均值，对分类特征进行编码的方法）
    A = X.to_numpy(copy=True)
    b = y.to_numpy(copy=True)
    A[np.ix_(train_mask, cat_features)] = enc.fit_transform(
        A[np.ix_(train_mask, cat_features)], b[train_mask]
    ) #对训练集中的分类特征进行编码
    A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(
        A[np.ix_(val_mask + test_mask, cat_features)]
    ) #对验证集和测试集中的分类特征进行编码
    A = A.astype(float)
    return pd.DataFrame(A, columns=X.columns)