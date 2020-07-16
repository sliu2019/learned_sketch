import argparse
import os
import torch
from data.ghg import getGHG
from data.gas import getGas
from data.electric import getElectric
from evaluate import getbest_regression
from pathlib import Path
import sys
import time
from misc_utils import *
import warnings
from tqdm import tqdm
import numpy as np


def make_parser_reg():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data", type=str, default="ghg", help="ghg|gas|electric")
    aa("--m", type=int, default=10, help="m for S")
    aa("--iter", type=int, default=1000, help="total iterations")

    aa("--random", default=False, action='store_true', help="don't learn S! Just compute error on random S")

    aa("--size", type=int, default=-1, help="dataset size")
    aa("--lr", type=float, default=1.0, help="learning rate for gradient descent")
    aa("--raw", dest='raw', default=False, action='store_true', help="generate raw?")
    aa("--bestonly", dest='bestonly', default=False, action='store_true', help="only compute best?")
    aa("--device", type=str, default="cuda:0")

    aa("--k_sparse", type=int, default=1, help="number of values in a column of S, sketching mat")
    aa("--num_exp", type=int, default=1, help="number of times to rerun the experiment (for avg'ing results)")
    aa("--bs", type=int, default=1, help="batch size")
    aa("--initalg", type=str, default="random", help="random|kmeans|lev|gs|lev_cluster|load")
    aa("--load_file", type=str, default="", help="if initalg=load, where to get S?")

    aa("--save_fldr", type=str,
       help="folder to save experiment results into; if None, then general folder")  # default: None
    aa("--save_file", type=str, help="append to runtype, if not None")

    aa("--S_init_method", type=str, default="pm1", help="pm1|gaussian|gaussian_pm1")
    return parser


if __name__ == '__main__':
    runtype = "train_regression"
    initalg_name2fn_dict = {"kmeans": init_w_kmeans, "lev": init_w_lev, "gs": init_w_gramschmidt,
                            "lev_cluster": init_w_lev_cluster, "load": init_w_load}

    parser = make_parser_reg()
    args = parser.parse_args()

    rawdir = "/home/me/research/big-regression/" if get_hostname() == "owner-ubuntu" else "/Users/me/research/big-regression/"
    rltdir = "/home/me/research/big-regression/" if get_hostname() == "owner-ubuntu" else "/Users/me/research/big-regression/"

    print(args)
    m = args.m

    if args.data == 'ghg':
        save_dir_prefix = os.path.join(rltdir, "rlt", "ghg")
    elif args.data == 'gas':
        save_dir_prefix = os.path.join(rltdir, "rlt", "gas")
    elif args.data == 'electric':
        save_dir_prefix = os.path.join(rltdir, "rlt", "electric")
    else:
        print("Wrong data option!")
        sys.exit()

    if args.save_file:
        runtype = runtype + "_" + args.save_file
    if args.save_fldr:
        save_dir = os.path.join(save_dir_prefix, args.save_fldr, args_to_fldrname(runtype, args))
    else:
        save_dir = os.path.join(save_dir_prefix, args_to_fldrname(runtype, args))

    best_fl_save_dir = os.path.join(save_dir_prefix, "best_files")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(best_fl_save_dir):
        os.makedirs(best_fl_save_dir)

    if (not args.bestonly) and (len(os.listdir(save_dir))):
        print("This experiment is already done! Now exiting.")
        sys.exit()

    lr = args.lr
    if args.data == "ghg":
        AB_train, AB_test, n, d_a, d_b = getGHG(args.raw, args.size, rawdir, 100)
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

    best_file = os.path.join(best_fl_save_dir, "N=" + str(args.size) + '_best')
    if (not os.path.isfile(best_file)) or args.raw:
        print("computing optimal least squares (linear) approximations", best_file)
        getbest_regression(A_train, B_train, A_test, B_test, best_file)

    best_train, best_test = torch.load(best_file)
    print("Best: %f , %f" % (best_train, best_test))
    # print("ln 125 in train_regression; check if best errors are properly loaded")
    # IPython.embed()

    start = time.time()
    print_freq = 50

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

        # Initialize sparsity pattern
        if args.initalg == "random":
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()

        elif args.initalg == "load":
            initalg = initalg_name2fn_dict[args.initalg]
            sketch_vector, sketch_value_cpu, active_ind = initalg(args.load_file, exp_num, n, m)
            sketch_value = sketch_value_cpu.detach().to(args.device)

        # Note: we sample with repeats, so you may map 1 row of A to <k_sparse distinct locations
        sketch_vector.requires_grad = False
        if args.initalg != "load":
            if args.S_init_method == "pm1":
                sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
            elif args.S_init_method == "gaussian":
                sketch_value = torch.from_numpy(np.random.normal(size=[args.k_sparse, n]).astype("float32")).to(
                    args.device)
                # sketch_value_cpu = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
            elif args.S_init_method == "gaussian_pm1":
                sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
                sketch_value = sketch_value + torch.from_numpy(
                    np.random.normal(size=[args.k_sparse, n]).astype("float32")).to(args.device)

        sketch_value.requires_grad = True

        for bigstep in tqdm(range(args.iter)):
            if (bigstep % 1000 == 0) and it_lr > 1:
                it_lr = it_lr * 0.3
            if bigstep > 200:
                it_print_freq = 200

            fp_start_time = time.time()

            batch_rand_ind = np.random.randint(0, high=N_train, size=args.bs)
            AM = A_train[batch_rand_ind].to(args.device)
            BM = B_train[batch_rand_ind].to(args.device)
            S = torch.zeros(m, n).to(args.device)
            S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(
                    args.k_sparse)] = sketch_value.reshape(-1)

            if bigstep % it_print_freq == 0 or bigstep == (args.iter - 1):
                # IPython.embed()
                train_err, test_err = save_iteration_regression(S, A_train, B_train, A_test, B_test, it_save_dir, bigstep)
                train_errs.append(train_err)
                test_errs.append(test_err)
                if bigstep == (args.iter - 1):
                    avg_over_exps += (test_err/args.num_exp)
                if args.random:
                    # don't train! after evaluating, exit trial
                    break

            SA = torch.matmul(S, AM)
            SB = torch.matmul(S, BM)
            U, Sig, V = torch.svd(SA)
            X = V.matmul(torch.diag_embed(1.0/Sig)).matmul(U.permute(0, 2, 1)).matmul(SB)
            ans = AM.matmul(X)
            loss = torch.mean(torch.norm(ans - BM, dim=(1,2)))

            # print("fp: ", time.time() -fp_start_time)
            fp_times.append(time.time() - fp_start_time)
            bp_start_time = time.time()
            loss.backward()
            # print("bp: ", time.time() -bp_start_time)
            bp_times.append(time.time() - bp_start_time)

            with torch.no_grad():
                if args.initalg == "load":
                    sketch_value[active_ind] -= (it_lr / args.bs) * sketch_value.grad[active_ind]
                    sketch_value.grad.zero_()
                else:
                    sketch_value -= (it_lr / args.bs) * sketch_value.grad
                    sketch_value.grad.zero_()

            del SA, SB, U, Sig, V, X, ans, loss
            torch.cuda.empty_cache()

        np.save(os.path.join(it_save_dir, "train_errs.npy"), train_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "test_errs.npy"), test_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "fp_times.npy"), fp_times, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "bp_times.npy"), bp_times, allow_pickle=True)
    print(avg_over_exps)