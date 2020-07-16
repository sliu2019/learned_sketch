import numpy as np
import os
import torch
import h5py
import IPython
import pandas as pd
def processRaw(N,rawdir,scale):
    dat_filenames = os.listdir(os.path.join(rawdir, "raw", "ghg_data"))

    count = 0
    for i in np.random.choice(np.arange(len(dat_filenames)), size=N, replace=False):
        print(count)
        fpath = os.path.join(rawdir, "raw", "ghg_data", dat_filenames[i])
        pd_dataframe = pd.read_csv(fpath, delim_whitespace=True)
        np_array = pd_dataframe.to_numpy()
        print(np_array.shape)
        print(dat_filenames[i])
        A = np_array[:-1].T
        AM = torch.from_numpy(A)
        U, Sig, V = torch.svd(AM)
        AM = AM/(Sig[0]/100)

        B = np_array[-1][:, None]
        BM = torch.from_numpy(B)
        U, Sig, V = torch.svd(BM)
        BM = BM/(Sig[0]/100)

        # print("ln 21 in ghg.py")
        # IPython.embed() # check shape
        if count == 0:
            A_train, A_test = torch.empty((0, A.shape[0], A.shape[1])), torch.empty((0, A.shape[0], A.shape[1]))
            B_train, B_test = torch.empty((0, B.shape[0], B.shape[1])), torch.empty((0, B.shape[0], B.shape[1]))
        if count < 0.8*N:
            # IPython.embed()
            A_train = torch.cat((A_train, AM[None].type(torch.float32)), dim=0)
            B_train = torch.cat((B_train, BM[None].type(torch.float32)), dim=0)
        else:
            A_test = torch.cat((A_test, AM[None].type(torch.float32)), dim=0)
            B_test = torch.cat((B_test, BM[None].type(torch.float32)), dim=0)
        count += 1

    # print("ln 28")
    # IPython.embed()
    torch.save([A_train, B_train], os.path.join(rawdir,"raw", "ghg", "train_"+str(N) + ".dat"))
    torch.save([A_test, B_test], os.path.join(rawdir,"raw", "ghg", "test_"+str(N) + ".dat"))

def getGHG(raw,N,rawdir,scale):
    if N<0:
        N=5
    if raw:
        processRaw(N,rawdir,scale)
    AB_train = torch.load(os.path.join(rawdir,"raw", "ghg", "train_"+str(N) + ".dat"))
    AB_test = torch.load(os.path.join(rawdir,"raw", "ghg", "test_"+str(N) + ".dat"))
    n = AB_train[0][0].size()[0]
    d_a = AB_train[0][0].size()[1]
    d_b = AB_train[1][0].size()[1]
    # 327 14 1
    return AB_train, AB_test, n, d_a, d_b
