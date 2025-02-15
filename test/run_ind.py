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
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices #tran(划分数据集方式不一样)

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]  #训练中能观察到的子集——观察子集（节点——节点特征、节点标签）
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        obs_g.create_formats_() 
        '''#tran没有，作用是为图对象创建不同的存储格式。
        COO格式适用于快速插入、删除和查找边，但在存储和处理大规模图时会占用更多的内存。
CSR和CSC格式适用于高效地进行图算法计算，如广度优先搜索、深度优先搜索等，但插入和删除边的效率较低。
        
        '''
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