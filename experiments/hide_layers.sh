#留到最后要加东西的时候跑
# for l in 1 3
# do
#     python train_teacher.py --exp_setting "tran" --teacher "SAGE" --dataset "cora" \
#                         --num_exp 1 --max_epoch 200 --patience 50\
#                         --num_layers $l \
#                         --hidden_dim 256
# done

# python train_teacher.py --exp_setting "tran" --teacher "MLP" --dataset "cora" \
#                     --num_exp 1 --max_epoch 200 --patience 50\
#                     --hidden_dim 1024\
#                     --num_layers 3


# python train_student.py --exp_setting "tran" --teacher "SAGE" --student "MLP" --dataset "cora" \
#                     --num_exp 1 --max_epoch 200 --patience 50\
#                     --hidden_dim 1024\
#                     --num_layers 3

# python train_student.py --exp_setting "tran" --teacher "SAGE" --student "MLP" --dataset "cora" \
#                     --num_exp 1 --max_epoch 200 --patience 50\
#                     --hidden_dim 2048\
#                     --num_layers 3
