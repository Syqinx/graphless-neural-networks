import numpy as np
import copy
import torch
import dgl
from utils import set_seed

"""
1. Train and eval
"""
def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """（训练教师模型时<tran或ind>除了MLP（train_mini_batch）和SAGE（train_sage）以外的模型训练）
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    (由run_transductive和run_inductive调用)
    参数：model-模型Model；data-整个图数据；feats-, labels, criterion-损失函数, optimizer, idx_train, lamb=1
    返回：loss_val-
    """
    model.train()
    '''是pytorch中一个特殊的方法，用于将模型设置为训练模式
    因为某些层（如Dropout和BatchNorm）在训练和评估（即，当模型用于预测，而不是训练时）模式下的行为是不同的
        1.训练模式下，Dropout层会随机丢弃一部分神经元（即，将它们的输出设置为0），以防止过拟合;
          BatchNorm层会使用当前batch的统计数据来进行标准化
        2.评估模式下，Dropout层会保留所有神经元，BatchNorm层会使用整个训练集的统计数据来进行标准化
    '''

    # Compute loss and prediction
    logits = model(data, feats) #将数据（data和feats）输入到模型中，并获取模型的输出
    out = logits.log_softmax(dim=1) 
    '''对模型输出结果进行log-softmax操作
    log-softmax则是在softmax的基础上再取对数（这通常用于多分类问题中，因为这样可以使得网络的输出可以直接用于计算交叉熵损失）
    【注】1.使用softmax层，转化为各个类别概率，这种情况，你应该紧接着使用CrossEntropyLoss损失函数；
          使用log_softmax层，那么使用NLLLOSS最好而且最直接。
          2.CrossEntropyLoss通常用于比较模型输出与真实标签；NLLLOSS通常用于比较模型输出与教师模型输出
    '''
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad() #用于清零优化器的梯度（PyTorch 中，当你调用 backward() 方法计算梯度时，梯度会被累积，而不是被替换）
    loss.backward() #自动计算 loss 关于模型参数的梯度，并将这些梯度存储在对应参数的 .grad 属性中
    optimizer.step() #根据之前计算的梯度来更新模型的参数
    return loss_val

def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb=1):  #lamb是用来平衡两个损失函数（此时只有教师模型，因此lamb=1）
    """SAGE模型训练，使用dataloder对图进行mini-batch处理，而不是整个图g
    lamb: weight parameter lambda
    参数：model, dataloader, feats, labels, criterion, optimizer, lamb=1
    返回：`total_loss/len(dataloader)`-
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader): #遍历dataloader(对dataloader中的每一batch数据进行以下处理)
        blocks = [blk.int().to(device) for blk in blocks] #将数据块转换为整数类型，并移动到特征数据所在的设备
        batch_feats = feats[input_nodes] 
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.用大数据集训练SAGE模型（以小批量处理数据 + 忽略整个图，只使用节点特征）
    lamb: weight parameter lambda
    参数：model, feats, labels-输入的out_t教师模型的软标签, batch_size-批次大小, criterion-评估器, optimizer, lamb=1
    返回：total_loss / num_batches —— 多批次训练的平均损失
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size) #批次数量
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1: #如果批次数量等于 1，则将idx_batch转换为1行，否则转换为num_batches行
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function
        logits = model(None, feats[idx_batch[i]]) #将输入数据（data和feats）传递给模型（model），并获取模型的输出（logits通常指的是网络的原始、未归一化的输出）
        out = logits.log_softmax(dim=1) #对logits进行log-softmax操作

        loss = criterion(out, labels[idx_batch[i]]) #计算损失
        total_loss += loss.item() #累加损失

        #计算损失、清零梯度、计算梯度并更新模型参数
        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches

def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data 所有输入数据的log概率
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx. 评估损失和分数（如果idx_eval不是None，则只在这些idx上评估损失和分数）
    参数：model, data, feats, labels, criterion, evaluator, idx_eval=None
    返回：out；loss.item()；score
    """
    model.eval() #将模型设置为评估模式
    with torch.no_grad(): #一个上下文管理，用于禁用自动梯度计算（因为在评估模型时通常不需要计算梯度）
        logits = model.inference(data, feats)  #将输入数据（data和feats）传递给模型（model），并获取模型的输出（logits通常指的是网络的原始、未归一化的输出）
        out = logits.log_softmax(dim=1)  #对logits进行log-softmax操作
        #如果idx_eval不是None，则只在这些idx上评估损失和分数，否则在所有数据上评估
        if idx_eval is None: 
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    return out, loss.item(), score

def evaluate_mini_batch(
    model, feats, labels, criterion, batch_size, evaluator, idx_eval=None
):
    """评估MLP模型（由run_transductive、run_inductive、distill_run_transductive、distill_run_inductive调用）
    参数：model-要评估的模型（本质MLP）, feats-特征矩阵, labels-标签矩阵, criterion, batch_size, evaluator, idx_eval=None
    返回：out-所有输入数据的log概率；loss & score (float)-评估的损失和分数，如果idx_eval不是None，则只在这些idx上评估损失和分数
    """
    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        for i in range(num_batches):
            logits = model.inference(None, feats[batch_size * i : batch_size * (i + 1)])
            out = logits.log_softmax(dim=1) #对预测结果进行 log_softmax 操作。这会将 logits 转换为概率分布，并对其进行对数转换。
            out_list += [out.detach()] #当前批次的输出添加到 out_list 中

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return out_all, loss.item(), score


"""
2. Run teacher
"""
def run_transductive(  
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
):
    """（由train_teacher.py调用)
    输入图默认很大，因此SAGE模型用于GNN，小批量用于MLP模型。
    参数：conf-配置；model-模型；g-DGLGraph对象；feats-特征矩阵；labels-标签矩阵；indices-指定训练、验证、测试集的索引；criterion-损失函数；evaluator-评估器；optimizer-优化器；logger-日志记录器；loss_and_score-损失和精确度存储(list)；
    返回：out-, score_val, score_test
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name: #如果是SAGE模型
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data = dataloader #创建一个用于训练的数据加载器（dataloader）。它使用了一个采样器（sampler，每层采样指定数量的邻居conf['fanout']）来采样节点的邻居
        data_eval = dataloader_eval #创建一个用于评估的数据加载器（dataloader_eval）。它使用了一个全邻居采样器（MultiLayerFullNeighborSampler、采集距离1跳的全部）来采样节点的邻居
    elif "MLP" in model.model_name: #如果是MLP模型
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
    else:
        g = g.to(device)
        data = g
        data_eval = g

    #初始化最佳的epoch、最佳的验证分数、计数器
    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name: #SAGE模型用train_sage函数训练
            loss = train_sage(model, data, feats, labels, criterion, optimizer)
        elif "MLP" in model.model_name: #MLP模型用train_mini_batch函数训练
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else: #其他模型用train函数训练（全图）
            loss = train(model, data, feats, labels, criterion, optimizer, idx_train)

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, labels_test, criterion, batch_size, evaluator
                )
            else:
                out, loss_train, score_train = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_train
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(out[idx_val], labels[idx_val]).item()
                score_val = evaluator(out[idx_val], labels[idx_val])
                loss_test = criterion(out[idx_test], labels[idx_test]).item()
                score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    score_train,
                    score_val,
                    score_test,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        out, _, score_val = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_val
        )
    else:
        out, _, score_val = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_val
        )

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test

def run_inductive(
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
):
    """（由train_teacher.py调用）

    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.
    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    参数：conf-模型配置,model-具体模型,g-图完整数据,feats-节点特征矩阵,labels-标签,indices,criterion,evaluator,optimizer,logger,loss_and_score,
    返回：
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"] #批处理大小
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        obs_dataloader = dgl.dataloading.NodeDataLoader(
            obs_g,
            obs_idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.NodeDataLoader(
            obs_g,
            torch.arange(obs_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        dataloader_eval = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(
                model, obs_data, obs_feats, obs_labels, criterion, optimizer
            )
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(
                model,
                obs_data,
                obs_feats,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
            )

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    labels_test_tran,
                    criterion,
                    batch_size,
                    evaluator,
                )
                _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    labels_test_ind,
                    criterion,
                    batch_size,
                    evaluator,
                )
            else:
                obs_out, loss_train, score_train = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    criterion,
                    evaluator,
                    obs_idx_train,
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(
                    obs_out[obs_idx_val], obs_labels[obs_idx_val]
                ).item()
                score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
                loss_test_tran = criterion(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                ).item()
                score_test_tran = evaluator(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                )

                # Evaluate the inductive part with the full graph
                out, loss_test_ind, score_test_ind = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
                )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_train,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]
            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        obs_out, _, score_val = evaluate_mini_batch(
            model, obs_feats, obs_labels, criterion, batch_size, evaluator, obs_idx_val
        )
        out, _, score_test_ind = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_test_ind
        )

    else:
        obs_out, _, score_val = evaluate(
            model,
            obs_data_eval,
            obs_feats,
            obs_labels,
            criterion,
            evaluator,
            obs_idx_val,
        )
        out, _, score_test_ind = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
        )

    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind


"""
3. Distill
"""
def distill_run_transductive(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """tran设置下的知识蒸馏训练和评估(由train_student.py的调用)
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: 教师模型输出的软标签
    criterion_l & criterion_t: 分别用于硬标签(labels)和软标签(out_t)的损失函数
    loss_and_score: 损失和得分的列表
    参数：conf-模型设置；model-；feats-特征矩阵；labels-标签向量；out_t_all-教师模型的输出结果；
    distill_indices-预先设置好的标签分组；criterion_l-用于硬标签的(将学生模型输出和真实标签比较)损失函数；criterion_t-用于软标签的(将学生模型输出和教师模型输出)损失函数；
    evaluator-评估器；optimizer-优化器；logger-日志记录器；loss_and_score-存储损失和分数的列表；
    返回：out-；score_val；score_test
    """
    #根据传入的参数设置变量
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    '''将数据都转移到GPU上：
    PyTorch中，可以使用.to(device)方法将数据从CPU转移到GPU。
    【注意】模型的参数和输入数据必须存储在同一个设备上，才能进行计算。也就是说，如果模型的参数存储在GPU上，那么输入数据也必须存储在GPU上。因此，在开始训练模型之前，通常需要先将数据从CPU转移到GPU。
    1.这里是将输入数据转移到GPU上
    2.另外在模型创建和初始化时，将模型参数转移到GPU上
    '''
    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)

    '''根据切割好的索引，划分各种数据'''
    feats_l, labels_l = feats[idx_l], labels[idx_l]
    feats_t, out_t = feats[idx_t], out_t_all[idx_t]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]

    #初始化最佳的epoch、最佳的验证分数、计数器
    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_mini_batch( #训练，返回硬标签损失
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb
        )
        loss_t = train_mini_batch( #训练，返回软标签损失
            model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb
        )
        loss = loss_l + loss_t

        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator
            )
            _, loss_test, score_test = evaluate_mini_batch(
                model, feats_test, labels_test, criterion_l, batch_size, evaluator
            )

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, _, score_val = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test

def distill_run_inductive(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_mini_batch(
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb
        )
        loss_t = train_mini_batch(
            model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator
            )
            _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
            )
            _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind
