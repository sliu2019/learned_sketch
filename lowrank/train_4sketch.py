import argparse
import os
import torch
from data.hyperspectra import getHyper
from data.tech import getTech
from data.videos import getVideos
from evaluate import evaluate,evaluate_both,getbest,evaluate_dense
from pathlib import Path
import sys
import time
from misc_utils import *
import warnings
from tqdm import tqdm
import numpy as np

def make_4sketch_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data", type=str, default="tech", help="tech|video|hyper|generate")
    aa("--dataname", type=str, default="mit", help="mit|eagle|friends")
    aa("--m", type=int, default=10, help="m for S")
    aa("--k", type=int, default=10, help="target: rank k approximation")
    aa("--m_r", type=int, default=10, help="m_r for R")
    aa("--m_t", type=int, default=10, help="left dim of T, countsketch")
    aa("--m_w", type=int, default=10, help="right dim of W, countsketch")
    aa("--iter", type=int, default=5000, help="total iterations")
    aa("--size", type=int, default=-1, help="dataset size")
    aa("--scale", type=int, default=100, help="scale")

    aa("--lr", type=float, help="gd learning rate")
    aa("--learn_R", dest='learn_R', default=False, action='store_true', help="is the R sketching matrix also learned?")
    aa("--learn_T", dest='learn_T', default=False, action='store_true', help="is the T sketching matrix also learned?")
    aa("--learn_W", dest='learn_W', default=False, action='store_true', help="is the W sketching matrix also learned?")

    aa("--save_fldr", type=str,
       help="folder to save experiment results into; if None, then general folder")
    aa("--save_file", type=str, help="append to runtype, if not None")

    aa("--k_sparse", type=int, default=1, help="number of values in a column of S, sketching mat")
    aa("--num_exp", type=int, default=1, help="number of times to rerun the experiment (for avg'ing results)")
    aa("--bs", type=int, default=1, help="batch size")
    aa("--initalg", type=str, default="random", help="random|kmeans|lev")
    aa("--bw", dest='bw', default=False, action='store_true', help="input images to black and white")
    aa("--dwnsmp", type=int, default=1, help="how much to downsample input images")
    aa("--single", dest='single', default=False, action='store_true', help="generate raw?")
    aa("--dense", type=int, default=-1, help="calculate dense?")
    # aa("--lr_S", type=float, default=1, help="learning rate scale?")
    aa("--raw", dest='raw', default=False, action='store_true', help="generate raw?")
    aa("--bestonly", dest='bestonly', default=False, action='store_true', help="only compute best?")
    aa("--device", type=str, default="cuda:0")
    return parser

if __name__ == '__main__':
    runtype = "4sketch"
    parser = make_4sketch_parser()
    initalg_name2fn_dict = {"kmeans": init_w_kmeans, "lev": init_w_lev}

    args = parser.parse_args()
    rawdir="/home/me/research/big-lowrank/" if get_hostname()=="owner-ubuntu" else "/Users/me/research/big-lowrank/"
    rltdir="/home/me/research/big-lowrank/" if get_hostname()=="owner-ubuntu" else "/Users/me/research/big-lowrank/"

    print(args)
    m=args.m
    k=args.k

    assert(args.m_t > args.m_r, "Choose m_t >> m_r")
    assert(args.m_w > args.m, "Choose m_w >> m_s")

    if args.data == 'tech':
        save_dir_prefix = rltdir + 'rlt/tech/'
    elif args.data == 'hyper':
        save_dir_prefix = rltdir + 'rlt/hyper/'
    elif args.data == 'video':
        save_dir_prefix = rltdir + 'rlt/video/' + args.dataname + '/'
    elif args.data == 'generate':
        save_dir_prefix = rltdir + 'rlt/generate'
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

    lr=1
    if args.data=='tech':
        A_train,A_test,n,d=getTech(args.raw,rawdir,args.scale)
    elif args.data=='hyper':
        A_train,A_test,n,d=getHyper(args.raw,args.size,rawdir,args.scale)
    elif args.data=='video':
        A_train,A_test,n,d=getVideos(args.dataname,args.raw,args.size,rawdir,args.scale, args.bw, args.dwnsmp)
        lr=10
    elif args.data == 'generate':
        shape = (k**2, 10*k)
        n = shape[0]
        d = shape[1]
        A = np.random.normal(size=(k, d))
        dup_rows = np.tile(A[-1], [n-k, 1])
        # IPython.embed()
        A = np.concatenate((A, dup_rows), axis=0)
        A_train = torch.from_numpy(A).type(torch.float32)[None, :,:].repeat(2, 1, 1)
        A_test = torch.from_numpy(A).type(torch.float32)[None,:,:]
        lr = 10

    if args.lr is not None:
        lr = args.lr

    print("Working on data ", args.data)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    N_train=len(A_train)
    N_test=len(A_test)
    print("Dim= ", n,d)
    print("N train=", N_train, "N test=", N_test)

    best_file = os.path.join(best_fl_save_dir, "N="+str(args.size)+"_k="+str(k)+'_best')
    if (not os.path.isfile(best_file)) or args.raw:
        print("computing optimal k-rank approximations ", best_file)
        getbest(A_train, A_test, k, args.data, best_file)

    best_train, best_test = torch.load(best_file)
    print("best: ", best_train, best_test)
    rlt_dic={}
    sparse=(args.data=='tech')
    if sparse:
        print("Not equipped to run on tech (sparse) data")
        raise(NotImplementedError)
        sys.exit()

    if args.dense>=0:
        for take in range(5):
            f_name ='m='+str(m)+'_k='+str(k)+'_N='+str(args.size)+'_full_take='+str(take)
            sketch = torch.randn(m, n).to(args.device)
            rlt_dic[f_name] = (evaluate_dense(sparse,A_train,sketch,m,k),
                               evaluate_dense(sparse,A_test,sketch,m,k))
            torch.save([rlt_dic[f_name], N_train, N_test], save_dir+f_name)
            print(f_name, rlt_dic[f_name][0]/N_train-best_train, rlt_dic[f_name][1]/N_test-best_test)
        if args.dense==1:
            sys.exit()

    start=time.time()

    print_freq = 50

    # save args
    args_save_fpath = os.path.join(save_dir, "args_it_0.pkl")
    f = open(args_save_fpath, "wb")
    pickle.dump(vars(args), f)
    f.close()

    m_r = args.m_r
    m_t = args.m_t
    m_w = args.m_w
    small_const = 1e-06
    for exp_num in range(args.num_exp):

        it_save_dir = os.path.join(save_dir,"exp_%d" % exp_num)
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
        elif args.data == 'generate' and args.initalg == "kmeans":
            sketch_vector = torch.from_numpy(np.append(np.arange(k), (k - 1) * np.ones(shape[0] - k)))
        else:
            # IPython.embed()
            if args.k_sparse != 1:
                print("Did not implement fancy sketch_vector initialization for k_sparse > 1")
                raise(NotImplementedError)
            print("creating sketch_vector")
            initalg = initalg_name2fn_dict[args.initalg]
            sketch_vector = initalg(A_train, m, k)[None, :]

        # Note: we sample with repeats, so you may map 1 row of A to <k_sparse distinct locations
        sketch_vector.requires_grad = False
        sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
        sketch_value.requires_grad = True

        R_sketch_vector = torch.randint(m_r, [d]).int()
        R_sketch_vector.requires_grad = False
        R_sketch_value = ((torch.randint(2, [d]).float() - 0.5) * 2).to(args.device)
        R_sketch_value.requires_grad = args.learn_R

        T_sketch_vector = torch.randint(args.m_t, [n]).int()
        T_sketch_vector.requires_grad = False
        T_sketch_value = ((torch.randint(2, [n]).float() - 0.5) * 2).to(args.device)
        T_sketch_value.requires_grad = args.learn_T

        W_sketch_vector = torch.randint(args.m_w, [d]).int()
        W_sketch_vector.requires_grad = False
        W_sketch_value = ((torch.randint(2, [d]).float() - 0.5) * 2).to(args.device)
        W_sketch_value.requires_grad = args.learn_W


        avg_detailed_timing = np.zeros((5))
        for bigstep in tqdm(range(args.iter)):
            # start = time.time()
            if (bigstep%1000==0) and it_lr>1:
                it_lr=it_lr*0.3
            if bigstep>200:
                it_print_freq=200
            A = A_train[np.random.randint(0, high=N_train, size=args.bs)]

            AM = A.to(args.device)
            Ad=d
            An=n

            S = torch.zeros(m, n).to(args.device)
            S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(args.k_sparse)] = sketch_value.reshape(-1)

            R = torch.zeros(d, m_r).to(args.device)
            R[torch.arange(d), R_sketch_vector.type(torch.LongTensor).reshape(-1)] = R_sketch_value.reshape(-1)

            T = torch.zeros(m_t, n).to(args.device)
            T[T_sketch_vector.type(torch.LongTensor), torch.arange(n)] = T_sketch_value

            W = torch.zeros(d, m_w).to(args.device)
            W[torch.arange(d), W_sketch_vector.type(torch.LongTensor)] = W_sketch_value
            if bigstep % it_print_freq == 0 or bigstep == (args.iter - 1):
                train_err, test_err = save_iteration_4sketch(S, R, T, W, A_train, A_test, args, it_save_dir, bigstep)
                train_errs.append(train_err)
                test_errs.append(test_err)

            fp_start_time = time.time()  # forward pass timing

            time_dict = []
            temp_start = time.time()
            AR = AM.matmul(R)
            SA = S.matmul(AM)
            TAR = T.matmul(AR)
            TAW = T.matmul(AM).matmul(W)
            SAW = SA.matmul(W)

            C = TAR
            D = SAW
            G = TAW
            time_dict.append(time.time() - temp_start)
            temp_start = time.time()

            U_c, sig_c, V_c = torch.svd(C)
            U_d, sig_d, V_d = torch.svd(D)
            R_c = torch.diag_embed(sig_c).matmul(V_c.permute(0, 2, 1))
            R_d = torch.diag_embed(sig_d).matmul(U_d.permute(0, 2, 1))
            U_d = V_d

            time_dict.append(time.time() - temp_start)

            temp_start = time.time()
            G_proj = (U_c.permute(0, 2, 1)).matmul(G).matmul(U_d)
            U1, Sig1, V1 = torch.svd(G_proj)
            X_prime_L = U1[:,:,:k].matmul(torch.diag_embed(Sig1[:,:k]))
            X_prime_R = V1.permute(0, 2, 1)[:,:k]
            time_dict.append(time.time() - temp_start)

            temp_start = time.time()
            rk_c = R_c.shape[1]
            T_c = R_c[:, :, :rk_c]
            T_c_inv = torch.inverse(T_c)
            top = T_c_inv.matmul(X_prime_L)
            bottom = torch.zeros(args.bs, m_r - rk_c, k).to(args.device)
            X_L = torch.cat((top, bottom), dim=1)

            rk_d = R_d.shape[1]
            T_d = R_d[:, :, :rk_d]
            T_d_inv = torch.inverse(T_d)
            left = X_prime_R.matmul(T_d_inv.permute(0, 2, 1))
            right = torch.zeros(args.bs, k, m - rk_d).to(args.device)
            X_R = torch.cat((left, right), dim=2)
            time_dict.append(time.time() - temp_start)

            temp_start = time.time()
            X = X_L.matmul(X_R)
            ans = AR.matmul(X).matmul(SA)

            loss = torch.mean(torch.norm(ans-AM, dim=(1,2)))
            time_dict.append(time.time() - temp_start)

            fp_time = time.time() - fp_start_time
            fp_times.append(fp_time)
            bp_start_time = time.time() # backwards pass timing
            loss.backward()
            bp_time = time.time() -bp_start_time
            bp_times.append(bp_time)

            with torch.no_grad():
                # IPython.embed()
                sketch_value -= (it_lr / args.bs) * sketch_value.grad
                sketch_value.grad.zero_()
                if args.learn_R:
                    R_sketch_value -= (it_lr / args.bs) * R_sketch_value.grad
                    R_sketch_value.grad.zero_()
                if args.learn_T:
                    T_sketch_value -= (it_lr / args.bs) * T_sketch_value.grad
                    T_sketch_value.grad.zero_()
                if args.learn_W:
                    W_sketch_value -= (it_lr / args.bs) * W_sketch_value.grad
                    W_sketch_value.grad.zero_()

            del C, D, G, U_c, R_c, U_d, R_d, G_proj, U1, Sig1, V1, X_prime_L, X_prime_R, rk_c, T_c, T_c_inv, top, bottom, X_L, rk_d, T_d, T_d_inv, left, right, X_R, X, ans, loss, S, R, T, W, AR, SA, TAR, TAW, SAW, A, AM
            torch.cuda.empty_cache()

        np.save(os.path.join(it_save_dir, "train_errs.npy"), train_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "test_errs.npy"), test_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "fp_times.npy"), fp_times, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "bp_times.npy"), bp_times, allow_pickle=True)