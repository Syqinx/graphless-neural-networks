#!/bin/bash

# Train SAGE teacher on "ogbn-arxiv"

for e in "tran" "ind"
do
    # num_exp=10
    # python train_teacher.py --exp_setting $e --teacher "SAGE" --dataset "ogbn-arxiv" \
    #                         --num_exp 10 --max_epoch 200 --patience 50 \
    #                         --save_results

    # num_exp=1
    python train_teacher.py --exp_setting $e --teacher "SAGE" --dataset "ogbn-arxiv" \
                            --num_exp 1 --max_epoch 200 --patience 50 \
                            --save_results
done
