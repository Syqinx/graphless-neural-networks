#MLP
# for e in "tran" "ind"
# do
#     for ds in "cora" "citeseer" "pubmed" "a-computer" "a-photo" "ogbn-arxiv" "ogbn-products"
#     do
#         python train_teacher.py --exp_setting $e --teacher "MLP" --dataset $ds \
#                             --num_exp 1 --max_epoch 200 --patience 50
#     done
# done

# MLP+(orgb-arxiv)
for e in "tran" "ind"
do
    python train_teacher.py --exp_setting $e --teacher "MLP3w4" --dataset "ogbn-arxiv" \
                        --num_exp 1 --max_epoch 200 --patience 50
done

#MLP+(orgb-products)
for e in "tran" "ind"
do
    python train_teacher.py --exp_setting $e --teacher "MLP3w8" --dataset "ogbn-products" \
                        --num_exp 1 --max_epoch 200 --patience 50
done
