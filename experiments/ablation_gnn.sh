#!/bin/bash


# Train five different teachers "GCN" "GAT" "SAGE" "MLP" "APPNP"
# on five datasets: "cora" "citeseer" "pubmed" "a-computer" "a-photo"
# Then train corresponding GLNN for each teacher
:<<!
消融实验2：
    分别在tran和ind设置下，不同模型在不同数据集上训练 + 不同教师模型在不同数据集上训练蒸馏glnn
!

aggregated_result_file="ablation_gnn.txt"
printf "Teacher\n" >> $aggregated_result_file    

for e in "tran" #"ind"
do
    printf "%6s\n" $e >> $aggregated_result_file
    for t in "GCN" "GAT" "APPNP" #"SAGE" "MLP"
    do
        printf "%6s\n" $t >> $aggregated_result_file
        for ds in "cora" #"citeseer" "pubmed" "a-computer" "a-photo"
        do
            # printf "%10s\t" $ds >> $aggregated_result_file
            # python train_teacher.py --exp_setting $e --teacher $t --dataset $ds --num_exp 5 \
            #                         --max_epoch 200 --patience 50 >> $aggregated_result_file
            printf "%10s\t" $ds >> $aggregated_result_file
            python train_teacher.py --exp_setting $e --teacher $t --dataset $ds --num_exp 1 \
                                    --max_epoch 200 --patience 50 >> $aggregated_result_file
        done
        printf "\n" >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file    
done

printf "Student\n" >> $aggregated_result_file    

for e in "tran" #"ind"
do
    printf "%6s\n" $e >> $aggregated_result_file
    for t in "GCN" "GAT" "APPNP" #"SAGE"
    do
        printf "%6s\n" $t >> $aggregated_result_file
        for ds in "cora" #"citeseer" "pubmed" "a-computer" "a-photo"
        do
            # printf "%10s\t" $ds >> $aggregated_result_file
            # python train_student.py --exp_setting $e --teacher $t --dataset $ds --num_exp 5 \
            #                         --max_epoch 200 --patience 50 >> $aggregated_result_file
            printf "%10s\t" $ds >> $aggregated_result_file
            python train_student.py --exp_setting $e --teacher $t --dataset $ds --num_exp 1 \
                                    --max_epoch 200 --patience 50 >> $aggregated_result_file
        done  #学生模型默认是MLP
        printf "\n" >> $aggregated_result_file
    done
    printf "\n" >> $aggregated_result_file    
done
