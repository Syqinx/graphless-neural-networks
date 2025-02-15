#torch
# import torch
# if torch.cuda.is_available():
#     print("CUDA is available")
# else:
#     print("CUDA is not available")

# utils.get_training_config
# import utils
# utils.get_training_config('./train.conf.yaml', 'GCN', 'cora')

#dataloader.load_penn94_mat
# import dataloader
# dataloader.load_penn94_mat('./data/penn94.mat')

#deepwalk
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dgl.data import CoraGraphDataset
from dgl.sampling import random_walk
from gensim.models import Word2Vec

# 1. 创建图
dataset = CoraGraphDataset()
# print('dataset: ', dataset) #CoraGraphDataset(n_graphs=1)
# print(dir(dataset))
# print(dataset[1])
g = dataset[0] #dataset[i]返回数据集中第i个图的图数据（Cora数据集中只有一个图，自然只能获取dataset[0]）
# print('g: ', g) 
'''Graph(num_nodes=2708, num_edges=10556, ndata_schemes={'train_mask': Scheme(shape=(), dtype=torch.bool), 'label': Scheme(shape=(), dtype=torch.int64), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool), 'feat': Scheme(shape=(1433,), dtype=torch.float32)}, edata_schemes={})
'''
# print('type(g)' ,type(g)) #<class 'dgl.heterograph.DGLHeteroGraph'>

# 2. 随机游走生成节点序列
walks,_ = random_walk(g, g.nodes(), length=10)
walks = walks.tolist()
walks = [[str(node) for node in walk] for walk in walks]

# 3. 使用 Word2Vec 学习节点向量表示
model = Word2Vec(walks, size=64, window=5, min_count=0, sg=1, workers=4)

# 获取节点向量表示
embeddings = [model.wv[str(i)] for i in range(g.number_of_nodes())]
embeddings = torch.tensor(embeddings)
# print(embeddings.shape) #torch.Size([2708, 64])
