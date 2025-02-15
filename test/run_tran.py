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

    idx_train, idx_val, idx_test = indices ##ind(划分数据集方式不一样)

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