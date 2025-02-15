#!/bin/bash

# Train GLNN with SAGE teacher on five datasets: "cora" "citeseer" "pubmed" "a-computer" "a-photo"
# 以SAGE为教师模型，分别在五个数据集上和"tran"和"ind"设置下训练模型

aggregated_result_file="glnn_cpf.txt"  #保存训练结果
for e in "tran" "ind"
do
    printf "%6s\n" $e >> $aggregated_result_file  # 将e值写入aggregated_result_file文件，%6s表示一个宽度为6的字符串
    for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo"
    do
        printf "%10s\t" $ds >> $aggregated_result_file
        # python train_student.py --exp_setting $e --teacher "SAGE" --dataset $ds --num_exp 10 \
        #                         --max_epoch 200 --patience 50 \
        #                         --save_results >> $aggregated_result_file
        python train_student.py --exp_setting $e --teacher "SAGE" --dataset $ds --num_exp 1 \
                                --max_epoch 200 --patience 50 \
                                --save_results >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file    
done
