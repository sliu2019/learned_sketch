import numpy as np
import os
import torch
import h5py
import IPython
import pandas as pd
import math
def processRaw(rawdir, scale):
    df = pd.read_csv(os.path.join(rawdir, "raw", "LD2011_2014.txt"), sep=";")
    data = df.iloc[:, 1:]
    data = data.applymap(lambda x: float(str(x).replace(",",".")))
    data = data.to_numpy()

    # dates = df.iloc[:, 0]
    # dates = dates.to_numpy()

    m = 95
    d_a = 1000
    d_b = 50
    A_combo = torch.empty((0, m, d_a))
    B_combo = torch.empty((0, m, d_b))

    for i in range(data.shape[1]):
        data_i = data[:, i]

        nzro = np.nonzero(data_i)[0]
        x = nzro.size
        if x > 100000:
            start_ind = (nzro[0]//95)*95
            A = data_i[start_ind: start_ind + (m*d_a)]
            A = A.reshape(m, d_a, order='F')
            AM = torch.from_numpy(A)
            U, Sig, V = torch.svd(AM)
            AM = AM / (Sig[0] / 100)

            B = data_i[start_ind + (m*d_a): start_ind + m*(d_a + d_b)]
            B = B.reshape(m, d_b, order='F')
            BM = torch.from_numpy(B)
            U, Sig, V = torch.svd(BM)
            BM = BM / (Sig[0] / 100)

            A_combo = torch.cat((A_combo, AM[None].type(torch.float32)), dim=0)
            B_combo = torch.cat((B_combo, BM[None].type(torch.float32)), dim=0)


    N = A_combo.size()[0]
    split_pt = math.ceil(N*0.8)
    A_train = A_combo[:split_pt]
    A_test = A_combo[split_pt:]
    B_train = B_combo[:split_pt]
    B_test = B_combo[split_pt:]

    torch.save([A_train, B_train], os.path.join(rawdir,"raw", "gas", "train_"+str(N) + ".dat"))
    torch.save([A_test, B_test], os.path.join(rawdir,"raw", "gas", "test_"+str(N) + ".dat"))
    return N

def getElectric(raw,rawdir,scale):
    if raw:
        N = processRaw(rawdir,scale)

    N = 313
    # print("Set N as variable here")
    # IPython.embed()
    AB_train = torch.load(os.path.join(rawdir,"raw", "gas", "train_"+str(N) + ".dat"))
    AB_test = torch.load(os.path.join(rawdir,"raw", "gas", "test_"+str(N) + ".dat"))
    n = AB_train[0][0].size()[0]
    d_a = AB_train[0][0].size()[1]
    d_b = AB_train[1][0].size()[1]
    # IPython.embed()
    # 950 50 50
    return AB_train, AB_test, n, d_a, d_b
