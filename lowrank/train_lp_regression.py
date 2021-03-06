import argparse
import os
import torch
from data.ghg import getGHG
from data.gas import getGas
from data.electric import getElectric
from evaluate import *
#from torchviz import make_dot, make_dot_from_trace
from pathlib import Path
import sys
import time
import math
import random
from misc_utils import *
import warnings
from tqdm import tqdm
import numpy as np

from scipy.stats import levy_stable


def make_parser_reg():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data", type=str, default="gas", help="ghg|gas|electric")
   # aa("--dataname", type=str, default="mit", help="eagle|mit|friends")
    aa("--m", type=int, default=10, help="m for S")
    aa("--iter", type=int, default=10000, help="total iterations")
    # aa("--scale", type=int, default= 100, help="scale") # not a functioning argument, assume 100

    aa("--random", default=False, action='store_true',
       help="don't learn S! Just compute error on random S")

    aa("--size", type=int, default=2900, help="dataset size")
    aa("--lr", type=float, default=5e-2, help="learning rate for gradient descent")
    aa("--raw", dest='raw', default=True,
       action='store_true', help="generate raw?")
    aa("--bestonly", dest='bestonly', default=False,
       action='store_true', help="only compute best?")
    aa("--device", type=str, default="cuda:0")

    # aa("--n_sample_rows", type=int, default=-1, help="Train with n_sample_rows rows")
    aa("--k_sparse", type=int, default=1,
       help="number of values in a column of S, sketching mat")
    aa("--num_exp", type=int, default=1,
       help="number of times to rerun the experiment (for avg'ing results)")
    aa("--bs", type=int, default=64, help="batch size")
    aa("--initalg", type=str, default="random",
       help="random|kmeans|lev|gs|lev_cluster|load")
    aa("--load_file", type=str, default="",
       help="if initalg=load, where to get S?")

    aa("--save_fldr", type=str,
       help="folder to save experiment results into; if None, then general folder")  # default: None
    aa("--save_file", type=str, help="append to runtype, if not None")

    aa("--S_init_method", type=str, default="gaussian",
       help="pm1|gaussian|gaussian_pm1")
    aa("--greedy_number", type=int, default=3,
       help="the number of sampling")
    return parser


if __name__ == '__main__':
    runtype = "train_regression_lp"
    parser = make_parser_reg()
    args = parser.parse_args()
    rawdir = "/home/lynette"
    rltdir = "/home/lynette"
    print(args)
    m = args.m

    if args.data == 'ghg':
        save_dir_prefix = os.path.join(rltdir, "rlt", "ghg+LP")
        # print("---------------testing1-----------")
        # TODO
    elif args.data == 'gas':
        save_dir_prefix = os.path.join(rltdir, "rlt", "gas+LP")
    elif args.data == 'electric':
        save_dir_prefix = os.path.join(rltdir, "rlt", "electric+LP")
    else:
        print("Wrong data option!")
        sys.exit()

    if args.save_file:
        runtype = runtype + "_" + args.save_file
    if args.save_fldr:
        save_dir = os.path.join(
            save_dir_prefix, args.save_fldr, args_to_fldrname(runtype, args))
    else:
        save_dir = os.path.join(
            save_dir_prefix, args_to_fldrname(runtype, args))
    # print("---------------testing2-----------")
    best_fl_save_dir = os.path.join(save_dir_prefix, "best_files")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(best_fl_save_dir):
        os.makedirs(best_fl_save_dir)

    if (not args.bestonly) and (len(os.listdir(save_dir))):
        print("This experiment is already done! Now exiting.")
        # sys.exit()
    lr = args.lr  # default = 1
    if args.data == "ghg":
        AB_train, AB_test, n, d_a, d_b = getGHG(
            args.raw, args.size, rawdir, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]
    elif args.data == "gas":
        AB_train, AB_test, n, d_a, d_b = getGas(args.raw, rawdir, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]
    elif args.data == "electric":
        AB_train, AB_test, n, d_a, d_b = getElectric(args.raw, rawdir, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]
    print("Working on data ", args.data)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    N_train = len(A_train)
    N_test = len(A_test)
    print("Dim= ", n, d_a, d_b)
    print("N train=", N_train, "N test=", N_test)
    p = 1.5
    best_file = os.path.join(best_fl_save_dir, "N=" + str(args.size) + '_best')
    print("computing best lp regression loss")
    getbest_lp_regression(A_train, B_train, A_test, B_test, best_file, p)
    best_train, best_test = torch.load(best_file)
    print("Best: train:%f , test:%f" % (best_train, best_test))

    start = time.time()
    print_freq = 50  # TODO

    # save args
    args_save_fpath = os.path.join(save_dir, "args_it_0.pkl")
    f = open(args_save_fpath, "wb")
    pickle.dump(vars(args), f)
    f.close()

    avg_over_exps = 0
    for exp_num in range(args.num_exp):

        it_save_dir = os.path.join(save_dir, "exp_%d" % exp_num)
        it_print_freq = print_freq
        it_lr = lr

        if not os.path.exists(it_save_dir):
            os.makedirs(it_save_dir)

        test_errs = []
        train_errs = []
        fp_times = []
        bp_times = []
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        # Initialize sparsity pattern
        if args.initalg == "random":
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
            print()
            print("-----------doing random-----------")
        elif args.initalg == "load":
            # TODO: not implemented for ksparse
            initalg = initalg_name2fn_dict[args.initalg]
            sketch_vector, sketch_value_cpu, active_ind = initalg(
                args.load_file, exp_num, n, m)
            sketch_value = sketch_value_cpu.detach().cpu()

        # Note: we sample with repeats, so you may map 1 row of A to <k_sparse distinct locations
        sketch_vector.requires_grad = False
        if args.initalg != "load":
            if args.S_init_method == "pm1":
                sketch_value = (
                    (torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).cpu()
                # print("-------pm1----sketch_value--------")
                # print(sketch_value.size)
            elif args.S_init_method == "gaussian":
                sketch_value = torch.from_numpy(levy_stable.rvs(
                    p, 0, size=[args.k_sparse, n]).astype("float32")).cpu()
                # sketch_value_cpu = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
            elif args.S_init_method == "gaussian_pm1":
                sketch_value = (
                    (torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).cpu()
                sketch_value = sketch_value + torch.from_numpy(
                    np.random.normal(size=[args.k_sparse, n]).astype("float32")).cpu()
        ###########TO BE CHANGED###########
        greedy_fl_save_dir = os.path.join(save_dir_prefix, "greedy_matrix")
        greedy_file = os.path.join(
            greedy_fl_save_dir, "N=" + str(args.size) + '_greedy')
        if not os.path.exists(greedy_fl_save_dir):
            os.makedirs(greedy_fl_save_dir)
        best_temp = 0
        if (not os.path.isfile(greedy_file)):
            S_add = torch.zeros(m, n).cpu()
            S = torch.zeros(m, n).cpu()
            S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(
                args.k_sparse)] = sketch_value.reshape(-1).cpu()
            S_original = S_add + S
            del S_add
            print("now calculating the greedy init S")
            best_temp = evaluate_to_rule_them_all_lp_regression(
                A_train, B_train, S, p)
            for i in range(S.shape[1]):
                random_vector = torch.tensor(
                    random.sample(range(0, m), args.greedy_number))
                random_value = torch.from_numpy(np.random.random_sample(
                    args.greedy_number,).astype("float32")*4-2)
                loc = sketch_vector[0][i]
                val = sketch_value[0][i]
                for j in range(random_vector.shape[0]):
                    for k in range(random_value.shape[0]):
                        S_temp = S+S_add
                        for l in range(m):
                            S_temp[l][i] = 0
                        S_temp[random_vector[j]][i] = random_value[k]
                        now_evaluate = evaluate_to_rule_them_all_lp_regression(
                            A_train[0:1], B_train[0:1], S_temp, p)
                        if now_evaluate < best_temp:
                            loc = random_vector[j]
                            val = S_temp[loc][i]
                            S[loc][i] = val
                            best_temp = now_evaluate
                sketch_value[0][i] = val
                sketch_vector[0][i] = loc
                if i % 50 == 0:
                    print("initing greedily S ", i/n*100, "%")
                random_train = evaluate_to_rule_them_all_lp_regression(
                    A_train[0:1], B_train[0:1], S_original, p)
            torch.save([S_original, random_train, best_temp,
                        sketch_value, sketch_vector], greedy_file)
        S_original, random_train, best_temp, sketch_value, sketch_vector = torch.load(
            greedy_file)
        print("before greedy initing,best_train:",
              random_train)
        print("After greedy initing, best_train:", best_temp)
        sketch_value.requires_grad = True
        ###########TO BE CHANGED###########
        for bigstep in tqdm(range(args.iter)):
            if (bigstep % 1000 == 0) and it_lr > 1:
                it_lr = it_lr * 0.1
            if bigstep > 200:
                it_print_freq = 200

            fp_start_time = time.time()
            # to randomly choose for gd
            batch_rand_ind = np.random.randint(0, high=N_train, size=args.bs)

            AM = A_train[batch_rand_ind].cpu()
            BM = B_train[batch_rand_ind].cpu()

            S = torch.zeros(m, n).cpu()
            S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(
                args.k_sparse)] = sketch_value.reshape(-1)

            if bigstep % 10 == 0 or bigstep == (args.iter - 1):
                train_err, test_err = save_iteration_lp_regression(
                    S, A_train, B_train, A_test, B_test, it_save_dir, bigstep, p)
                train_errs.append(train_err)
                test_errs.append(test_err)
                if bigstep == (args.iter - 1):
                    # num_exp =how many times run
                    avg_over_exps += (test_err/args.num_exp)
                if args.random:
                    # don't train! after evaluating, exit trial
                    break
            SA = torch.matmul(S, AM)
            SB = torch.matmul(S, BM)
            X = lp_norm_regression(p, SA, SB)
            ans = AM.matmul(X.float())
            loss = torch.mean(torch.norm(abs(ans - BM), dim=(1, 2), p=p))
            fp_times.append(time.time() - fp_start_time)
            bp_start_time = time.time()
            loss.backward()
            bp_times.append(time.time() - bp_start_time)
            # TODO: Maybe don't have to divide by args.bs: is this similar to lev_score_experiments bug?
            # However, if you change it, then you need to compensate in lr... all old exp will be invalidated
            with torch.no_grad():
                if args.initalg == "load":
                    sketch_value[active_ind] -= (it_lr / args.bs) * \
                        sketch_value.grad[active_ind]
                    sketch_value.grad.zero_()
                else:
                    sketch_value -= (it_lr / args.bs) * sketch_value.grad
                    sketch_value.grad.zero_()

            # del SA, SB, U, Sig, V, X, ans, loss
            del SA, SB, X, ans, loss
            torch.cuda.empty_cache()

        np.save(os.path.join(it_save_dir, "train_errs.npy"),
                train_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "test_errs.npy"),
                test_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "fp_times.npy"),
                fp_times, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "bp_times.npy"),
                bp_times, allow_pickle=True)
    print(avg_over_exps)
