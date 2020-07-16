# MRR SGD
python -u train_regression.py --data electric --m 10 --num_exp 5 --save_fldr regression_ablation --device cuda:0 --bs 20 --lr 30.000000 --iter 1000 --bestonly

# MRR greedy
python -u regression_init_utils.py --row_order random --data ghg --m 20 --num_exp 3 --n_early_factor 1.000000 --save_fldr greedy_ablation_random --device cuda:0

# Lp regression SGD
python -u train_lp_regression.py --data electric --m 10 --num_exp 1 --save_fldr regression_ablation --device cuda:1 --bs 20 --lr 30.000000 --iter 1000 --bestonly --random --raw

# Huber regression SGD
python -u train_huber_regression.py --data electric --m 10 --num_exp 1 --save_fldr regression_ablation --device cuda:1 --bs 20 --lr 30.000000 --iter 1000 --bestonly --random --raw

# 1-sketch LRA SGD
python -u train_speedup_direct.py --iter 1000 --num_exp 3 --lr 30.000000 --bs 12 --data video --dataname mit --m 60 --k 30 --size 500 --save_fldr rsketch_ablation --device cuda:0 --bestonly

# 1-sketch LRA greedy
python -u sparsity_pattern_init_algs.py --row_order dec_row_norm --data video --dataname mit --m 20 --k 20 --num_exp 3 --n_early_factor 1.000000 --save_fldr greedy_ablation_redo_sh --device cuda:1

# 4-sketch LRA SGD
python -u train_4sketch.py --m_t 140 --m_r 130 --m 60 --m_w 70 --learn_R --learn_T --learn_W --lr 10.0 --iter 1000 --num_exp 3 --bs 1 --data video --dataname friends --k 30 --size 500 --save_fldr 4sketch_ablation --device cuda:1 --bestonly

