import numpy as np
import os
import torch
import h5py
import IPython
import pandas as pd
def processRaw(rawdir, scale):
    # (13530, 11)
    filenames = os.listdir(os.path.join(rawdir, "raw", "dataset_twosources_raw"))
    sorted_fnms = sorted(filenames, key=lambda x: x[4:])
    # sorted_fnms = filenames
    ind_begin_test_data = 144
    min = 13530
    for i in range(len(filenames)):
        print(i)
        fpath = os.path.join(rawdir, "raw", "dataset_twosources_raw", sorted_fnms[i])
        pd_dataframe = pd.read_csv(fpath)
        np_array = pd_dataframe.to_numpy()
        # print(np_array.shape)
        # print(dat_filenames[i])

        A = np_array[:min, -5:]
        AM = torch.from_numpy(A)
        U, Sig, V = torch.svd(AM)
        AM = AM/(Sig[0]/100)

        B = np_array[:min, 5:]
        BM = torch.from_numpy(B)
        U, Sig, V = torch.svd(BM)
        BM = BM/(Sig[0]/100)

        # A = np_array[:min//2]
        # B = np_array[min//2:min]
        # print("ln 21 in ghg.py")
        # print(np_array.shape)
        # if np_array.shape[0] < min:
        #     min = np_array.shape[0]
        # IPython.embed() # check shape
        if i == 0:
            A_train, A_test = torch.empty((0, A.shape[0], A.shape[1])), torch.empty((0, A.shape[0], A.shape[1]))
            B_train, B_test = torch.empty((0, B.shape[0], B.shape[1])), torch.empty((0, B.shape[0], B.shape[1]))
        if i < ind_begin_test_data:
            # IPython.embed()
            # print("hi")
            A_train = torch.cat((A_train, AM[None].type(torch.float32)), dim=0)
            B_train = torch.cat((B_train, BM[None].type(torch.float32)), dim=0)
        else:
            # print("hey")
            A_test = torch.cat((A_test, AM[None].type(torch.float32)), dim=0)
            B_test = torch.cat((B_test, BM[None].type(torch.float32)), dim=0)
    # print(min)
    # print("ln 28")
    # IPython.embed()
    N = 180
    torch.save([A_train, B_train], os.path.join(rawdir,"raw", "gas", "train_"+str(N) + ".dat"))
    torch.save([A_test, B_test], os.path.join(rawdir,"raw", "gas", "test_"+str(N) + ".dat"))

def getGas(raw,rawdir,scale):
    if raw:
        processRaw(rawdir,scale)
    N = 180
    AB_train = torch.load(os.path.join(rawdir,"raw", "gas", "train_"+str(N) + ".dat"))
    AB_test = torch.load(os.path.join(rawdir,"raw", "gas", "test_"+str(N) + ".dat"))
    n = AB_train[0][0].size()[0]
    d_a = AB_train[0][0].size()[1]
    d_b = AB_train[1][0].size()[1]
    # IPython.embed()
    # 13530 5 6
    return AB_train, AB_test, n, d_a, d_b
