import numpy as np
import os
import torch
import h5py
import IPython
def processRaw(N,rawdir,scale):
    # A_train=[]
    # A_test=[]
    for i in range(1,N):
        print(i)
        fname=rawdir+'raw/HS-SOD/hyperspectral/'+str(i).zfill(4)+'.mat'
        if os.path.exists(fname):
            f = h5py.File(fname, 'r')
            AList=f['hypercube'][:]
            for j in range(AList.shape[0]):
                As = torch.from_numpy(AList[j]).view(AList[j].shape[0], -1).float()
                # print(As.size())
                U, S, V = As.svd()
                div = abs(S[0].item())
                if div < 1e-10:
                    div = 1
                    print("Catch!")
                div /= scale
                # if np.random.random()<0.8:
                #     A_train.append(As/div)
                # else:
                #     A_test.append(As/div)

                As = (As/div).float()
                if i == 1:
                    A_train, A_test = torch.empty((0, As.size()[0], As.size()[1])), torch.empty(
                        (0, As.size()[0], As.size()[1]))
                if np.random.random() < 0.8:
                    A_train = torch.cat((A_train, As[None]), dim=0)
                else:
                    A_test = torch.cat((A_test, As[None]), dim=0)
    # print("ln 28")
    # IPython.embed()
    torch.save(A_train,rawdir+"raw/hyper_train_"+str(N)+"_"+str(scale)+".dat")
    torch.save(A_test,rawdir+"raw/hyper_test_"+str(N)+"_"+str(scale)+".dat")

def getHyper(raw,N,rawdir,scale):
    if N<0:
        N=5
    if raw:
        processRaw(N,rawdir,scale)
    # A_train,A_test=torch.load(rawdir+"raw/hyper_"+str(N)+"_"+str(scale)+".dat")
    A_train = torch.load(rawdir+"raw/hyper_train_"+str(N)+"_"+str(scale)+".dat")
    A_test = torch.load(rawdir+"raw/hyper_test_"+str(N)+"_"+str(scale)+".dat")
    # IPython.embed()
    return A_train,A_test, A_train.size()[1], A_train.size()[2]
