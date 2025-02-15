#!/bin/bash

# Train 1-hop GA-MLP and 1-hop GA-GLNN with SAGE teacher on "ogbn-arxiv" under the inductive setting
:<<!
具有一跳邻居特征增强的 GLNN
!

#先运行train_teacher.py，再运行train_student.py
#1.先单独使用MLP模型
python train_teacher.py --exp_setting "ind" --teacher "MLP3w4" --dataset "ogbn-arxiv" \
                        --num_exp 5 --max_epoch 200 --patience 50 \
                        --feature_aug_k 1


#2.再使用SAGE模型作为MLP的教师模型，两次实验做对比（在此之前需要运行对应的SAGE-ogbn-arxiv）
python train_student.py --exp_setting "ind" --teacher "SAGE" --student "MLP3w4" --dataset "ogbn-arxiv" \
                    --num_exp 5 --max_epoch 200 --patience 50 \
                    --feature_aug_k 1

