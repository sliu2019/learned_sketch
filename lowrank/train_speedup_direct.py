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

def make_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data", type=str, default="tech", help="tech|video|hyper|generate")
    aa("--dataname", type=str, default="mit", help="eagle|mit|friends")
    aa("--m", type=int, default=10, help="m for S")
    aa("--k", type=int, default=10, help="target: rank k approximation")
    # aa("--mp", type=int, default=10, help="mp for R")
    aa("--iter", type=int, default=1000, help="total iterations")
    aa("--size", type=int, default=-1, help="dataset size")
    # aa("--scale", type=int, default= 100, help="scale") # not a functioning argument, assume 100

    aa("--n_sample_rows", type=int, default=-1, help="Train with n_sample_rows rows")
    aa("--k_sparse", type=int, default=1, help="number of values in a column of S, sketching mat")
    aa("--num_exp", type=int, default=1, help="number of times to rerun the experiment (for avg'ing results)")
    aa("--bs", type=int, default=1, help="batch size")
    aa("--initalg", type=str, default="random", help="random|kmeans|lev|gs|lev_cluster|load")
    aa("--load_file", type=str, default="", help="if initalg=load, where to get S?")

    aa("--save_fldr", type=str,
       help="folder to save experiment results into; if None, then general folder")  # default: None
    aa("--save_file", type=str, help="append to runtype, if not None")

    aa("--S_init_method", type=str, default="pm1", help="pm1|gaussian|gaussian_pm1")
    aa("--lev_ridge", dest='lev_ridge', default=False, action='store_true',
       help="use ridge regression version with lambda?")
    aa("--lev_cutoff", type=int, help="how many top k to isolate; must be <=m? if m, then not isolate, but share")
    aa("--lev_count", default=False, action="store_true", help="use counting method to compute top k over A_train?")

    aa("--bw", dest='bw', default=False, action='store_true', help="input images to black and white")
    aa("--dwnsmp", type=int, default=1, help="how much to downsample input images")
    aa("--random", default=False, action='store_true', help="don't learn S! Just compute error on random S")
    aa("--dense", type=int, default=-1, help="calculate dense?")
    # aa("--lr_S", type=float, default=1.0, help="learning rate scale?")
    aa("--lr", type=float, help="learning rate for GD")
    aa("--raw", dest='raw', default=False, action='store_true', help="generate raw?")
    aa("--bestonly", dest='bestonly', default=False, action='store_true', help="only compute best?")
    aa("--device", type=str, default="cuda:0")
    return parser

if __name__ == '__main__':
    runtype = "train_direct_grad"
    initalg_name2fn_dict = {"kmeans": init_w_kmeans, "lev": init_w_lev, "gs": init_w_gramschmidt,
                            "lev_cluster": init_w_lev_cluster, "load": init_w_load}
    parser = make_parser()

    args = parser.parse_args()
    rawdir="/home/me/research/big-lowrank/" if get_hostname()=="owner-ubuntu" else "/Users/me/research/big-lowrank/"
    rltdir="/home/me/research/big-lowrank/" if get_hostname()=="owner-ubuntu" else "/Users/me/research/big-lowrank/"

    print(args)
    m=args.m
    # mp=args.mp
    k=args.k

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
        A_train,A_test,n,d=getTech(args.raw,rawdir,100)
        # print("ln 99 in train.py")
        # IPython.embed()
    elif args.data=='hyper':
        A_train,A_test,n,d=getHyper(args.raw,args.size,rawdir,100)
    elif args.data=='video':
        A_train,A_test,n,d=getVideos(args.dataname,args.raw,args.size,rawdir,100, args.bw, args.dwnsmp)
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

    print("LR: %f" % lr)
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
    print("Best: %f , %f" % (best_train, best_test))

    rlt_dic={}
    sparse=(args.data=='tech')
    if sparse:
        device = "cpu"  # won't fit on GPU after 4 iterations...

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
    print_freq = 50  # TODO

    # save args
    args_save_fpath = os.path.join(save_dir, "args_it_0.pkl")
    f = open(args_save_fpath, "wb")
    pickle.dump(vars(args), f)
    f.close()

    if args.n_sample_rows > 0 and args.n_sample_rows <= n:
        #numpy.random.choice(a, size=None, replace=True, p=None)¶
        sample_rows_inds = np.random.choice(np.arange(n), size=args.n_sample_rows, replace=False)

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
        train_start = time.time()

        # Initialize sparsity pattern
        if args.initalg == "random":
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
        elif args.data == 'generate' and args.initalg == "kmeans":
            sketch_vector = torch.from_numpy(np.append(np.arange(k), (k - 1) * np.ones(shape[0] - k)))
        else:
            if args.k_sparse != 1:
                print("Did not implement fancy sketch_vector initialization for k_sparse > 1")
                raise(NotImplementedError)
            print("creating sketch_vector")
            initalg = initalg_name2fn_dict[args.initalg]
            if args.initalg == "gs":
                sketch_vector = initalg(A_train, m, k, args.size, best_fl_save_dir)[None, :]
            elif args.initalg == "lev" or args.initalg == "lev_cluster":
                sketch_vector = initalg(A_train, m, k, args.lev_ridge, args.lev_cutoff, args.lev_count)[None, :]
            elif args.initalg == "load":
                sketch_vector, sketch_value_cpu, active_ind = initalg(args.load_file, exp_num, n, m)
                sketch_value = sketch_value_cpu.detach().to(args.device)
            else:
                sketch_vector = initalg(A_train, m, k)[None, :]

        # Note: we sample with repeats, so you may map 1 row of A to <k_sparse distinct locations
        sketch_vector.requires_grad = False
        if args.initalg != "load":
            if args.S_init_method == "pm1":
                sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
            elif args.S_init_method == "gaussian":
                sketch_value = torch.from_numpy(np.random.normal(size=[args.k_sparse, n]).astype("float32")).to(args.device)
                # sketch_value_cpu = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
            elif args.S_init_method == "gaussian_pm1":
                sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
                sketch_value = sketch_value + torch.from_numpy(np.random.normal(size=[args.k_sparse, n]).astype("float32")).to(args.device)

        sketch_value.requires_grad = True

        for bigstep in tqdm(range(args.iter)):
            # with autograd.detect_anomaly():
            if (bigstep%1000==0) and it_lr>1:
                it_lr=it_lr*0.3
            if bigstep>200:
                it_print_freq=200

            if sparse:
                assert(args.bs == 1, "If dataset = tech, then args.bs = 1")
                A = A_train[np.random.randint(0, high=N_train)]
                AM=A['M'][None].to(args.device)
                Ad=A['d']
                An=A['n']
                AMap=A['Map']
            else:
                A = A_train[np.random.randint(0, high=N_train, size=args.bs)]
                AM = A.to(args.device)
                Ad=d
                An=n

            if sparse:
                S = torch.zeros(m, n).to(args.device)
                S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(
                    args.k_sparse)] = sketch_value.reshape(-1)
                ind = torch.tensor(AMap).type(torch.LongTensor).to(args.device)
                S_sel = torch.index_select(S, dim=1, index=ind)
                SA = S_sel.matmul(AM)
            else:
                S = torch.zeros(m, n).to(args.device)
                if args.n_sample_rows >= m and args.n_sample_rows <= n:
                    zero_ind = sketch_vector.type(torch.LongTensor).reshape(-1)[sample_rows_inds]
                    zero_ind[:m] = torch.arange(m)
                    S[zero_ind, sample_rows_inds] = sketch_value.reshape(-1)[sample_rows_inds]
                else:
                    S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(args.k_sparse)] = sketch_value.reshape(-1)
                SA = torch.matmul(S, AM)

            # TODO
            if bigstep % it_print_freq == 0 or bigstep == (args.iter - 1):
                train_err, test_err = save_iteration(S, A_train, A_test, args, it_save_dir, bigstep, sparse=sparse)
                train_errs.append(train_err)
                test_errs.append(test_err)

                if args.random:
                    # don't train! after evaluating, exit trial
                    break
            fp_start_time = time.time()
            U2, Sigma2, V2 = torch.svd(SA) # returns compact SVD
            AU = AM.matmul(V2)
            U3, sigma3, V3 = torch.svd(AU)
            Sigma3 = torch.diag_embed(sigma3[:, :k]).to(args.device)
            ans = U3[:, :, :k].matmul(Sigma3).matmul(
                V3.permute(0, 2, 1)[:, :k]).matmul(V2.permute(0, 2, 1))
            loss = torch.mean(torch.norm(ans - AM, dim=(1, 2)))

            fp_times.append(time.time() - fp_start_time)
            bp_start_time = time.time()
            loss.backward()
            bp_times.append(time.time() - bp_start_time)

            # TODO: Maybe don't have to divide by args.bs: is this similar to lev_score_experiments bug?
            # However, if you change it, then you need to compensate in lr... all old exp will be invalidated
            with torch.no_grad():
                if args.initalg == "load":
                    sketch_value[active_ind] -= (it_lr/args.bs)*sketch_value.grad[active_ind]
                    sketch_value.grad.zero_()
                else:
                    """if torch.isnan(sketch_value.grad).any():
                        IPython.embed()"""
                    sketch_value -= (it_lr/args.bs)*sketch_value.grad
                    sketch_value.grad.zero_()

            del SA, U2, Sigma2, V2, AU, U3, Sigma3, V3, ans, loss, AM
            if sparse:
                del S_sel, ind
            torch.cuda.empty_cache()

        print(np.mean(np.array(fp_times)))
        np.save(os.path.join(it_save_dir, "train_errs.npy"), train_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "test_errs.npy"), test_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "fp_times.npy"), fp_times, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "bp_times.npy"), bp_times, allow_pickle=True)

        total_train_time = time.time() - train_start
        np.save(os.path.join(it_save_dir, "total_train_time.npy"), np.array([total_train_time]), allow_pickle=True)
