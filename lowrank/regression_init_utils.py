import numpy as np, torch, IPython, os
from data.videos import *
from global_variables import *
from tqdm import tqdm
# import tensorflow as tf
import time
import argparse
from data.hyperspectra import getHyper
from data.tech import getTech
import math
import sys
import pickle
from data.ghg import getGHG
from data.gas import getGas
from data.electric import getElectric
from evaluate import getbest_regression, evaluate_to_rule_them_all_regression
from misc_utils import get_hostname, return_data_fldr_pth

"""
Utilities for initializing sparsity patterns
"""

def evaluate(A_set, B_set, S, device):
    """
    BATCHED, but also iterative (i.e. for data=hyper, eval list may be ~3000)
    :param A: list of matrices (3D tensor)
    :param sketch: S or [S, S2] concatenated; assumed matrices
    :param k: low-rank k
    :return: K-rk approx cost, averaged over matrices in eval_list
    """
    n = A_set.size()[0]
    bs = 100
    loss = 0

    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i * bs:min(n, (i + 1) * bs)].to(device)
        BM = B_set[i * bs:min(n, (i + 1) * bs)].to(device)

        SA = torch.matmul(S, AM)
        SB = torch.matmul(S, BM)
        U, Sig, V = torch.svd(SA)

        Sig_np = Sig.cpu().numpy()
        Sig_inv_np = np.divide(1.0, Sig_np, out=np.zeros_like(Sig_np), where=(Sig_np != 0))
        Sig_inv = torch.from_numpy(Sig_inv_np).to(device)
        X = V.matmul(torch.diag_embed(Sig_inv)).matmul(U.permute(0, 2, 1)).matmul(SB)
        ans = AM.matmul(X)
        it_loss = torch.sum(torch.norm(ans - BM, dim=(1, 2)))/n
        loss += it_loss.item()

        del AM, BM, SA, SB, U, Sig, V, X, ans, it_loss
    return loss


def fast_rank1_update_svd(U, Sig, V, a, b, device):
    """
	Batched!
	Only need to compute V'
	inputs should all be on cuda/GPU
	:param U: bsxmxm
	:param Sig: bsxmxm
	:param V: bsxdxm
	:param a: (m*num_gs_samples) x m x 1
	:param b: (m*num_gs_samples) x d x 1
	:return: V'
	"""

    m = U.size()[0]
    rank = Sig.shape[0]

    m_tens = U[None].permute(0, 2, 1).matmul(a)
    p = a - U[None].matmul(m_tens)  # a perp U
    R_a = torch.norm(p, dim=1)

    R_a_np = R_a.cpu().numpy()
    R_a_inv_np = np.divide(1.0, R_a_np, out=np.zeros_like(R_a_np), where=R_a_np != 0)
    R_a_inv = torch.from_numpy(R_a_inv_np).to(device)
    P = p * R_a[:, :, None]

    n = V[None].permute(0, 2, 1).matmul(b)
    q = b - V[None].matmul(n)
    R_b = torch.norm(q, dim=1)
    Q = q * (1.0 / R_b[:, :, None])

    S_ext = torch.zeros(rank + 1, rank + 1).to(device)
    S_ext[:rank, :rank] = torch.diag(Sig)
    y = torch.cat((n, R_b[:, :, None]), dim=1)
    K = S_ext + torch.cat((m_tens, R_a[:, :, None]), dim=1).matmul(y.permute(0, 2, 1))

    bs = a.size()[0]

    V_tiled = V[None].repeat(bs, 1, 1)
    V_ext = torch.cat((V_tiled, Q), dim=2)

    u1, s1, v1 = torch.svd(K)
    V_prime = V_ext.matmul(v1)  # anyways, s1[m] is tiny

    U_tiled = U[None].repeat(bs, 1, 1)
    U_ext = torch.cat((U_tiled, P), dim=2)
    U_prime = U_ext.matmul(u1)

    del m_tens, p, R_a, P, n, q, R_b, Q, S_ext, y, K, V_tiled, V_ext, u1, v1
    return U_prime, s1, V_prime


def args_to_fldrname(args, parser, big_reg_fldr_pth):
    """
    :param args: from parse_args(), a namespace
    :return: str, foldername
    """
    ignore_keys = ["save_fldr", "save_file", "device", "data", "num_exp"]
    d_args = vars(args)
    exp_fldr = ""
    for key in sorted(d_args.keys()):
        if key not in ignore_keys and d_args[key] != parser.get_default(key):
            exp_fldr += "_" + str(key) + "_" + str(d_args[key])
    exp_fldr = exp_fldr[1:]
    exp_fldr_pth = os.path.join(big_reg_fldr_pth, "greedy", args.data, "gs", args.save_fldr, exp_fldr)
    return exp_fldr_pth


def fast_loss(bin_samples, gs_samples, S, AM, BM, i, device):
    """
	:param gs_samples:
	:param AM: n x d
	:param U0: m x m
	:param Sig0: m x m
	:param V0: d x m
	:param m:
	:param n:
	:param d:
	:param num_bin_samples:
	:return:
	"""
    m = S.shape[0]
    num_gs_samples = gs_samples.size()[0]
    num_bin_samples = bin_samples.size

    a = torch.zeros((num_bin_samples, m))
    a[np.arange(num_bin_samples), bin_samples] = 1.0
    a = a[:, :, None]
    a = torch.repeat_interleave(a, num_gs_samples, dim=0)
    a = a.to(device)

    # (m*num_gs_samples) x d x 1
    b = (gs_samples[:, None].matmul(AM[i][None]))[:, :, None]
    b = b.repeat(num_bin_samples, 1, 1)
    b = b.to(device)

    SA = S.matmul(AM)
    U0, Sig0, V0 = torch.svd(SA)
    U, Sig, V = fast_rank1_update_svd(U0, Sig0, V0, a, b, device)
    rank = Sig0.shape[0]
    V = V[:, :, :rank]
    U = U[:, :, :rank]
    Sig = Sig[:, :rank]
    a_cpu = a.cpu()
    del U0, Sig0, V0, a, b

    total = num_bin_samples * num_gs_samples
    gpu_bs = 100
    all_losses = torch.empty(total)

    # Prepare by computing SB
    SB = S.matmul(BM)
    b = (gs_samples[:, None].matmul(BM[i][None]))[:, :, None]
    b = b.repeat(num_bin_samples, 1, 1)
    b = b.permute(0, 2, 1)
    b_cpu = b.cpu()
    del b

    for j in range(math.ceil(total / float(gpu_bs))):
        start_ind = j * gpu_bs
        end_ind = min(total, (j + 1) * gpu_bs)
        V_batch = V[start_ind: end_ind]
        U_batch = U[start_ind: end_ind]
        Sig_batch = Sig[start_ind: end_ind]

        Sig_batch_np = Sig_batch.cpu().numpy()
        Sig_inv_np = np.divide(1.0, Sig_batch_np, out=np.zeros_like(Sig_batch_np), where=(Sig_batch_np!=0))
        Sig_inv = torch.from_numpy(Sig_inv_np).to(device)
        mp_pseudo = V_batch.matmul(torch.diag_embed(Sig_inv)).matmul(U_batch.permute(0, 2, 1))
        a_batch = a_cpu[start_ind:end_ind].to(device)
        b_batch = b_cpu[start_ind: end_ind].to(device)
        SB_batch = SB[None] + a_batch.matmul(b_batch)
        X = mp_pseudo.matmul(SB_batch)

        err = AM.matmul(X) - BM
        losses = torch.norm(err, dim=(1, 2))
        all_losses[start_ind: end_ind] = losses
        del V_batch, U_batch, Sig_batch, mp_pseudo, a_batch, b_batch, SB_batch, X, err, losses

    losses = all_losses
    losses[losses!=losses] = float("inf") # taking care of nan
    return losses


def init_w_greedy(A_train, B_train, A_test, B_test, m, n_early_factor, num_gs_samples, num_bin_samples, num_A, row_order, num_init_entries,
                  num_exp, device, args, exp_fldr_pth):
    """
    row_order: "random", "forwards", "backwards", "dec_row_norm", "lev_score"
    """
    # print(num_bin_samples)
    print("Running exp on device %s" % device)
    N_train = A_train.size()[0]
    n = A_train.size()[1]
    d_a = A_train.size()[2]
    d_b = B_train.size()[2]

    # early termination option
    end_ind = math.ceil(n * n_early_factor)

    # initialize variables
    if num_bin_samples is None:
        num_bin_samples = m

    if not os.path.exists(exp_fldr_pth):
        os.makedirs(exp_fldr_pth)

    # save arguments
    with open(os.path.join(exp_fldr_pth, "args.pkl"), 'wb') as handle:
        args_dict = vars(args)
        args_dict["n"] = n
        args_dict["d_a"] = d_a
        args_dict["d_b"] = d_b
        args_dict["end_ind"] = end_ind
        # print(num_init_entries)
        args_dict["num_init_entries"] = num_init_entries
        pickle.dump(args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for exp_num in range(num_exp):
        print("exp_num %d" % exp_num)
        save_path = os.path.join(exp_fldr_pth, "exp_%d" % exp_num)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Sample 1 training matrix
        rand_ind = np.random.randint(0, N_train)
        AM = (A_train[rand_ind]).to(device)
        AM_np = AM.data.cpu().numpy()
        BM = B_train[rand_ind].to(device)

        if row_order == "random":
            shuff_row_ind = np.arange(n)
            np.random.shuffle(shuff_row_ind)
        elif row_order == "forwards":
            shuff_row_ind = np.arange(n)
        elif row_order == "backwards":
            shuff_row_ind = np.arange(n)[::-1]
        elif row_order == "dec_row_norm":
            row_norms = np.linalg.norm(AM_np, axis=1)
            shuff_row_ind = np.argsort(row_norms)[::-1]
        elif row_order == "lev_score":
            U, Sig, VT = np.linalg.svd(AM_np)
            lev_scores = np.linalg.norm(U, axis=1)
            shuff_row_ind = np.argsort(lev_scores)[::-1]
        else:
            print("Invalid row order selection, exiting")
            sys.exit(0)

        sketch_vector = np.zeros(n)
        sketch_values = np.zeros(n).astype("float32")

        active_ind = shuff_row_ind[:num_init_entries]
        sketch_vector[active_ind] = np.arange(num_init_entries)
        sketch_values[active_ind] = ((np.random.randint(0, high=2, size=num_init_entries).astype("float32") - 0.5) * 2)

        count = 0
        print_freq = n // 10

        # init save data structs
        test_errs = []
        train_errs = []

        rand_inds = np.random.choice(np.arange(N_train), size=num_A, replace=False)
        for i in tqdm(shuff_row_ind[num_init_entries:end_ind]):
            if count > 200:
                print_freq = 200

            # sample
            gs_samples = torch.linspace(-2, 2, steps=num_gs_samples).to(
                device)
            if num_bin_samples == m:
                bin_samples = np.arange(m)
            else:
                bin_samples = np.random.choice(np.arange(m), size=num_bin_samples, replace=False)

            # create new S
            S = torch.zeros((m, n)).to(device)
            S[sketch_vector, torch.arange(n)] = torch.from_numpy(sketch_values).to(device)

            all_losses = torch.zeros((num_bin_samples*num_gs_samples))
            all_orders = np.zeros((num_bin_samples*num_gs_samples))
            if row_order == "random" and num_A > 1:
                for rand_ind in rand_inds:
                    # Sample 1 training matrix
                    AM = (A_train[rand_ind]).to(device)
                    BM = B_train[rand_ind].to(device)
                    losses = fast_loss(bin_samples, gs_samples, S, AM, BM, i, device)
                    all_losses += losses
                    all_orders += np.argsort(losses.numpy())
            else:
                all_losses = fast_loss(bin_samples, gs_samples, S, AM, BM, i, device)


            min_ind_flat = torch.argmin(all_losses)
            min_ind = [min_ind_flat // num_gs_samples, min_ind_flat % num_gs_samples]

            # update sketch vector/values
            sketch_vector[i] = torch.tensor(bin_samples[min_ind[0]])
            sketch_values[i] = gs_samples[min_ind[1]]
            active_ind = np.concatenate((active_ind, [i]))

            # every so often: evaluate (train and test) and save errors and sketch vector/values
            if count % print_freq == 0 or count == (end_ind - num_init_entries - 1):
                S = torch.zeros((m, n)).to(device)
                S[sketch_vector, torch.arange(n)] = torch.from_numpy(sketch_values).to(device)
                loss = evaluate(A_train[rand_inds], B_train[rand_inds], S, device)
                train_errs.append(loss)
                print("it %d, train errs: %f" % (count, loss))
                loss = evaluate(A_test, B_test, S, device)
                test_errs.append(loss)
                print("it %d, test errs: %f" % (count, loss))

                torch.save(
                    [torch.from_numpy(sketch_vector), torch.from_numpy(sketch_values), torch.from_numpy(active_ind)],
                    os.path.join(save_path, "saved_tensors_it_%d" % count))
                np.save(os.path.join(save_path, "train_errs.npy"), train_errs)
                np.save(os.path.join(save_path, "test_errs.npy"), test_errs)

            count += 1


def init_w_load(load_file, exp_num, m):
    """
	CAUTION: Should only be used for greedy experiments
	:param load_file:
	:param exp_num:
	:return: Expects everything to be torch tensor!
	"""
    big_lowrank_pth = "../big-regression"

    exp_args = pickle.load(open(os.path.join(big_lowrank_pth, load_file, "args.pkl"), "rb"))
    last_itr = exp_args["end_ind"] - m - 1

    full_flpth = os.path.join(big_lowrank_pth, load_file, "exp_%d" % exp_num, "saved_tensors_it_%d" % last_itr)
    if not os.path.exists(full_flpth):
        print(full_flpth, " does not exist")
        sys.exit(0)

    print("Loading pre-initialized sketch from %s" % full_flpth)
    x = torch.load(full_flpth)

    sketch_vector = x[0]
    sketch_value = x[1]
    active_ind = x[2]
    if type(sketch_vector) == np.ndarray:
        sketch_vector = torch.from_numpy(sketch_vector)

    return sketch_vector, sketch_value, active_ind


def create_parser_gs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data", type=str, default="ghg", help="ghg|gas|electric")
    aa("--m", type=int, default=None, help="m for S")

    aa("--num_exp", type=int, default=1, help="number of trials for this experiment")
    aa("--n_early_factor", type=float, default=None, help="Only place n*n_early_factor rows, instead of n")
    aa("--num_gs_samples", default=10, type=int, help="number of samples in the range [-2, 2] for the row weights")

    aa("--num_bin_samples", type=int, default=None, help="None means all bins; else sample fewer")
    aa("--row_order", type=str, default=None, help="random, forward, backward, lev_score, dec_row_norm")

    aa("--save_fldr", type=str, default="", help="describe what kind of exp this is")
    aa("--device", type=str, default="cuda:0", help="can set gpu per experiment")
    aa("--num_A", type=int, default=1, help="Number of A to average over")
    aa("--num_init_entries", type=int, default=1, help="Number of rows to initialize S with")
    return parser


if __name__ == "__main__":
    parser = create_parser_gs()
    args = parser.parse_args()

    big_reg_fldr_pth = return_data_fldr_pth("big-regression")

    raw = False
    if args.data == "ghg":
        size = 600
        AB_train, AB_test, n, d_a, d_b = getGHG(raw, size, big_reg_fldr_pth, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]
    elif args.data == "gas":
        AB_train, AB_test, n, d_a, d_b = getGas(raw, big_reg_fldr_pth, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]
    elif args.data == "electric":
        AB_train, AB_test, n, d_a, d_b = getElectric(raw, big_reg_fldr_pth, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]

    exp_fldr_pth = args_to_fldrname(args, parser, big_reg_fldr_pth)
    # print("check data loading and pathing")
    # IPython.embed()
    init_w_greedy(A_train, B_train, A_test, B_test, args.m, args.n_early_factor, args.num_gs_samples,
                  args.num_bin_samples, args.num_A, args.row_order, args.num_init_entries, args.num_exp, args.device, args, exp_fldr_pth)
