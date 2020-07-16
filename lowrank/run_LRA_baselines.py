import torch, numpy
import time
from datetime import datetime
import IPython
# from create_baseline_sketch import *
import sys, os, pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from global_variables import *
from misc_utils import *
import time

def evaluate_exact_SVD(A_train, A_test, k_m, dataset_spec, save_fldr_pth, num_trials, device):
    """
    """
    k = k_m[0]
    m = k_m[1]
    # Setting up saving
    if len(dataset_spec) == 1:
        exp_save_fldr_pth = os.path.join(save_fldr_pth, "k_%i_m_%i_dataset_%s_sketch_method_exact_SVD" % (k_m[0], k_m[1], dataset_spec[0]))
    else:
        exp_save_fldr_pth = os.path.join(save_fldr_pth, "k_%i_m_%i_dataset_%s_%s_sketch_method_exact_SVD" % (k_m[0], k_m[1], dataset_spec[0], dataset_spec[1]))

    print("Saving in %s" % exp_save_fldr_pth)
    if not os.path.exists(exp_save_fldr_pth):
        os.mkdir(exp_save_fldr_pth)

    # creating vars to be used later
    n_test = A_test.shape[0]
    n = A_test.shape[1]
    d = A_test.shape[2]
    n_train = A_train.shape[0]

    err_list = []
    ind_list = []
    runtime_list = []
    for i in range(num_trials):
        print(i)
        # Compute the sketch on A_train
        ind = np.random.randint(0, n_train)
        ind_list.append(ind)
        A_rand = A_train[ind].to(device)

        t_start = time.perf_counter()
        U, S, V = torch.svd(A_rand)
        V_m = V[:,:m]

        torch.save(V_m, os.path.join(exp_save_fldr_pth, "V_m_trial_%i" % i))
        # Compute the SVD loss on A_test
        AV = A_test.matmul(V_m[None])
        U2, s2, V2 = torch.svd(AV)
        S2 = torch.diag_embed(s2[:,:k]).to(device)
        approx = U2[:,:,:k].matmul(S2).matmul(V2.permute(0, 2, 1)[:, :k]).matmul(V_m.permute(1,0)[None])
        t_end = time.perf_counter()
        runtime_list.append((t_start - t_end))
        del A_rand, U, S, V, V_m, AV, U2, s2, S2, V2
        torch.cuda.empty_cache()

        approx_cpu = approx.cpu()
        A_test_cpu = A_test.cpu()
        del approx
        diff = approx_cpu - A_test_cpu
        loss = torch.mean(torch.norm(diff, dim=(1, 2)))
        print(loss.data.cpu())
        err_list.append(loss.data.cpu())
        del diff, loss
        torch.cuda.empty_cache()

        # saving
        np.save(os.path.join(exp_save_fldr_pth, "errs.npy"), np.array(err_list))
        np.save(os.path.join(exp_save_fldr_pth, "rand_indices.npy"), np.array(ind_list))

    print("Timing results: %f \pm %f" % (np.mean(runtime_list), np.std(runtime_list)))
    return np.mean(err_list), np.std(err_list)

def get_dataset(dataset_spec):
    raw = False
    size = 500
    bw = False
    dwnsmp = 1

    rawdir = "/home/me/research/big-lowrank/"
    dataset = dataset_spec[0]
    if dataset=='tech':
        A_train,A_test,n,d=getTech(raw,rawdir,100)
    elif dataset=='hyper':
        A_train,A_test,n,d=getHyper(raw,size,rawdir,100)
    elif dataset=='video':
        dataname = dataset_spec[1]
        A_train,A_test,n,d=getVideos(dataname,raw,size,rawdir,100, bw, dwnsmp)

    return A_train, A_test, n, d

if __name__ == "__main__":
    """
    Sketch methods: exact_SVD
    """
    LRA_fldr_pth = "/home/me/research/big-lowrank-baselines/"

    num_trials = 10
    save_folder_nm = "timing_test"
    device = "cuda:1"

    k_m_list = [(20, 20), (20, 40), (30, 30), (30, 60)]
    dataset_spec_list = [["video", "mit"], ["video", "eagle"], ["video", "friends"], ["hyper"]]
    sketch_method_list = ["exact_SVD"]

    # Save params
    save_fldr_pth = os.path.join(LRA_fldr_pth, save_folder_nm)
    if not os.path.isdir(save_fldr_pth):
        os.mkdir(save_fldr_pth)

    args = {"k_m_list": k_m_list, "dataset_spec_list": dataset_spec_list, "sketch_method_list": sketch_method_list, "num_trials": num_trials}
    with open(os.path.join(save_fldr_pth, 'args.pkl'), 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # create table
    table_pth = os.path.join(save_fldr_pth, "output_data.npy")
    table = np.empty((len(k_m_list), len(dataset_spec_list), len(sketch_method_list)), dtype="object")

    for j, dataset_spec in enumerate(dataset_spec_list):
        # Load data
        A_train, A_test, n, d = get_dataset(dataset_spec)

        A_test = A_test.to(device)
        for i, k_m in enumerate(k_m_list):
            for k, sketch_method in enumerate(sketch_method_list):
                if sketch_method == "exact_SVD":
                    result = evaluate_exact_SVD(A_train, A_test[[0]], k_m, dataset_spec, save_fldr_pth, num_trials, device)
                else:
                    print("sketch method not implemented, exiting")
                    sys.exit(0)

                # store in table
                print(result)
                table[i, j, k] = result

                # save table
                np.save(table_pth, table, allow_pickle=True)

