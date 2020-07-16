import torch
import numpy as np
import sys, os
from global_variables import *
import IPython
import math
from misc_utils import lp_norm_regression, huber_regression

################## For Huber regression ####################
def save_iteration_huber_regression(S, A_train, B_train, A_test, B_test, save_dir, bigstep, p):
    """
    Not implemented:
    Mixed matrix evaluation
    """
    torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)
    test_err = evaluate_to_rule_them_all_huber_regression(A_test, B_test, S)
    train_err = evaluate_to_rule_them_all_huber_regression(
        A_train, B_train, S)
    torch.save([[S], [train_err, test_err]], torch_save_fpath)

    print(train_err, test_err)
    print("Saved iteration: %d" % bigstep)
    return train_err, test_err


def evaluate_to_rule_them_all_huber_regression(A_set, B_set, S):
    n = A_set.size()[0]
    bs = 100
    loss = 0
    median = 0
    for i in range(math.ceil(n/float(bs))):
        #AM = A_set[i*bs:min(n, (i+1)*bs)].to(device)
        AM = A_set[i*bs:min(n, (i+1)*bs)].cpu()
        #BM = B_set[i*bs:min(n, (i+1)*bs)].to(device)
        BM = B_set[i*bs:min(n, (i+1)*bs)].cpu()

        #SA = torch.matmul(S.to(device), AM)
        SA = torch.matmul(S.cpu(), AM)

        #SB = torch.matmul(S.to(device), BM)
        SB = torch.matmul(S.cpu(), BM)

        #X = huber_regression(SA, SB)
        X = huber_regression(SA, SB).cpu()

        ans = AM.matmul(X.float())
        loss = abs(ans-BM)
        #median += np.median(loss.detach().numpy())
        median += np.median(loss.detach().cpu().numpy())
    median = median/math.ceil(n/float(bs))
    return median


def bestPossible_hb_regression(A_set, B_set):
    n = A_set.size()[0]
    bs = 100
    loss = 0
    median = 0
    for i in range(math.ceil(n/float(bs))):
        #AM = A_set[i*bs:min(n, (i+1)*bs)].to(device)
        AM = A_set[i*bs:min(n, (i+1)*bs)].cpu()
        #BM = B_set[i*bs:min(n, (i+1)*bs)].to(device)
        BM = B_set[i*bs:min(n, (i+1)*bs)].cpu()
        X = huber_regression(AM, BM)
        ans = AM.matmul(X.float())
        loss = abs(ans-BM)
        median += np.median(loss.detach().cpu().numpy())
    median = median/math.ceil(n/float(bs))
    # print("======================Huber regression===============")
    # print(median)
    # print("======================Huber regression===============")
    return median


def getbest_hb_regression(A_train, B_train, A_test, B_test, best_file):
    best_train_err = bestPossible_hb_regression(A_train, B_train)
    best_test_err = bestPossible_hb_regression(A_test, B_test)

    torch.save([best_train_err, best_test_err], best_file)
    return best_train_err, best_test_err
############################################################

################## For Lp regression ####################
def save_iteration_lp_regression(S, A_train, B_train, A_test, B_test, save_dir, bigstep, p):
    """
    Not implemented:
    Mixed matrix evaluation
    """
    torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)

    test_err = evaluate_to_rule_them_all_lp_regression(A_test, B_test, S, p)
    train_err = evaluate_to_rule_them_all_lp_regression(
        A_train, B_train, S, p)
    torch.save([[S], [train_err, test_err]], torch_save_fpath)

    print(train_err, test_err)
    print("Saved iteration: %d" % bigstep)
    return train_err, test_err


def evaluate_to_rule_them_all_lp_regression(A_set, B_set, S, p):
    n = A_set.size()[0]
    bs = 100
    loss = 0
    # median = 0
    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i * bs:min(n, (i + 1) * bs)].cpu()
        BM = B_set[i * bs:min(n, (i + 1) * bs)].cpu()
        SA = torch.matmul(S.cpu(), AM)
        SB = torch.matmul(S.cpu(), BM)
        X = lp_norm_regression(p, SA, SB).cpu()
        ans = AM.matmul(X.float())
        it_loss = torch.sum(torch.norm(abs(ans - BM), dim=(1, 2), p=p))//n
        loss += it_loss.item()
    return loss


def bestPossible_lp_regression(A_set, B_set, p):
    n = A_set.size()[0]
    bs = 100  # 100
    loss = 0
    median = 0
    A_set = A_set.cpu()
    B_set = B_set.cpu()
    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i * bs:min(n, (i + 1) * bs)].cpu()
        BM = B_set[i * bs:min(n, (i + 1) * bs)].cpu()
        X = lp_norm_regression(p, AM, BM).cpu()
        ans = AM.matmul(X.float())
        it_loss = torch.sum(torch.norm(abs(ans - BM), dim=(1, 2), p=p))//n
        loss += it_loss.item()
    print("======================LP regression===============")
    print(loss)
    print("======================LP regression===============")
    return loss

def getbest_lp_regression(A_train, B_train, A_test, B_test, best_file, p):
    best_train_err = bestPossible_lp_regression(A_train, B_train, p)
    best_test_err = bestPossible_lp_regression(A_test, B_test, p)

    torch.save([best_train_err, best_test_err], best_file)
    return best_train_err, best_test_err
#########################################################

def bestPossible(eval_list,k,data):
    totLoss = torch.tensor(0.0)
    for A in eval_list:
        print(".",end="")
        sys.stdout.flush()
        if data=='tech':
            AM=A['M'].to(device)
        else:
            AM=A.to(device)
        U, S, V = AM.svd()
        ans = U[:, :k].mm(torch.diag(S[:k]).to(device)).mm(V.t()[:k])
        # totLoss += torch.norm(ans - AM) ** 2
        totLoss += torch.norm(ans - AM)
    return totLoss

def evaluate(sparse, eval_list,sketch_vector, sketch_value,m,k,n,d):  # evaluate the test/train performance
    totLoss = 0
    count = 0


    for A in eval_list:
        if sparse:
            AM=A['M'].to(device)
            SA = torch.Tensor(m, A['d']).fill_(0).to(device)
            for i in range(A['n']):  # A has this many rows, not mapped yet
                actR = A['Map'][i]  # Actual row in the matrix
                mapR = sketch_vector[actR]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[actR]  # remember: times the weight
        else:
            AM=A.to(device)
            SA = torch.Tensor(m, d).fill_(0).to(device)
            for i in range(n):  # A has this many rows, not mapped yet
                mapR = sketch_vector[i]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight

        # print(SA.max().item(), SA.min().item(), SA.mean().item())
        # print(A.max().item(), A.min().item(), A.mean().item())

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        # totLoss += (torch.norm(ans - AM) ** 2).item()
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
        if (count % 10 == 0):
            print(count, end=",")
            sys.stdout.flush()
    return totLoss

def evaluate_dense(sparse, eval_list,sketch, m,k):  # evaluate the test/train performance
    totLoss = 0
    count = 0


    for A in eval_list:
        if sparse:
            AM=A['M'].to(device)
            SA = torch.Tensor(m, A['d']).fill_(0).to(device)
            for i in range(A['n']):  # A has this many rows, not mapped yet
                actR = A['Map'][i]  # Actual row in the matrix
                SA+=torch.ger(sketch[:,actR], AM[i])
        else:
            AM=A.to(device)
            SA=torch.mm(sketch, AM)

        # print(SA.max().item(), SA.min().item(), SA.mean().item())
        # print(A.max().item(), A.min().item(), A.mean().item())

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        # totLoss += (torch.norm(ans - AM) ** 2).item()
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
        if (count % 10 == 0):
            print(count, end=",")
            sys.stdout.flush()
    return totLoss


def evaluate_both(eval_list,sketch_vector, sketch_value,m,k,n,d):  # evaluate the test/train performance
    totLoss = 0
    count = 0

    for A in eval_list:
        if sparse:
            AM=A['M'].to(device)
            SA = torch.Tensor(m, A['d']).fill_(0).to(device)
            for i in range(A['n']):  # A has this many rows, not mapped yet
                actR = A['Map'][i]  # Actual row in the matrix
                mapR = sketch_vector[actR]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[actR]  # remember: times the weight
        else:
            AM=A.to(device)
            SA = torch.Tensor(m, d).fill_(0).to(device)
            for i in range(n):  # A has this many rows, not mapped yet
                mapR = sketch_vector[i]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
        if (count % 10 == 0):
            print(count, end=",")
            sys.stdout.flush()
    return totLoss
def evaluate_extra_dense(eval_list,sketch, sketch2, k):
    totLoss = 0
    count = 0
    for A in eval_list:
        AM=A.to(device)
        SA=torch.cat([torch.mm(sketch,AM),torch.mm(sketch2,AM)])

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
        # if (count % 10 == 0):
        #     print(count, end=",")
        #     sys.stdout.flush()
    return totLoss

def evaluate_to_rule_them_all_4sketch(A_set, S, R, T, W, k):
    S = S.cpu()
    R = R.cpu()
    T = T.cpu()
    W = W.cpu()

    n = A_set.shape[0]
    bs = 100
    loss = 0
    device = "cpu"
    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i*bs:min((i+1)*bs, n)]
        it_bs = min((i+1)*bs, n) - i*bs
        AR = AM.matmul(R)
        SA = S.matmul(AM)
        TAR = T.matmul(AR)
        TAW = T.matmul(AM).matmul(W)
        SAW = SA.matmul(W)

        """U1, Sig1, V1 = torch.svd(TAR)
        TAR_pinv = V1.matmul(torch.diag_embed(1.0 / Sig1)).matmul(U1.permute(0, 2, 1))
        U2, Sig2, V2 = torch.svd(SAW)
        SAW_pinv = V2.matmul(torch.diag_embed(1.0 / Sig2)).matmul(U2.permute(0, 2, 1))
        prod = TAR_pinv.matmul(TAW).matmul(SAW_pinv)
        U3, Sig3, V3 = torch.svd(prod)
        X = U3[:, :, :k].matmul(torch.diag_embed(Sig3[:, :k])).matmul(V3.permute(0, 2, 1)[:, :k])
        ans = AR.matmul(X).matmul(SA)
        it_loss = torch.sum(torch.norm(ans - AM, dim=(1, 2)))/n
        loss += it_loss.item()"""
        m_r = R.shape[1]
        m = S.shape[0]
        C = TAR
        D = SAW
        G = TAW

        # Full QR, not truncated
        U_c, R_c = torch.qr(C, some=True)
        U_d, R_d = torch.qr(D.permute(0, 2, 1), some=True)
        # print("ln 270")  # check that the dims of U_d, R_d are as expected
        # IPython.embed()

        G_proj = (U_c.permute(0, 2, 1)).matmul(G).matmul(U_d)
        U1, Sig1, V1 = torch.svd(G_proj)
        X_prime_L = U1[:, :, :k].matmul(torch.diag_embed(Sig1[:, :k]))
        X_prime_R = V1.permute(0, 2, 1)[:, :k]

        rk_c = R_c.shape[1]
        T_c = R_c[:, :, :rk_c]
        # T_c_safe = T_c + torch.eye(rk_c).to(args.device)*small_const
        # print("ln 279, check shape of X_primes")
        # IPython.embed()
        T_c_inv = torch.inverse(T_c)
        top = T_c_inv.matmul(X_prime_L)
        bottom = torch.zeros(it_bs, m_r - rk_c, k).to(device)
        X_L = torch.cat((top, bottom), dim=1)

        # check that c x_l = u_c x_l'!
        # IPython.embed()
        rk_d = R_d.shape[1]
        T_d = R_d[:, :, :rk_d]  # + torch.eye(rk_d).to(args.device)*small_const
        T_d_inv = torch.inverse(T_d)
        left = X_prime_R.matmul(T_d_inv.permute(0, 2, 1))
        right = torch.zeros(it_bs, k, m - rk_d).to(device)
        X_R = torch.cat((left, right), dim=2)

        # IPython.embed()
        X = X_L.matmul(X_R)
        ans = AR.matmul(X).matmul(SA)
        it_loss = torch.sum(torch.norm(ans - AM, dim=(1, 2)))/n
        loss += it_loss.item()

    return loss

def evaluate_to_rule_them_all_rsketch(A_set, S, R, k):
    S = S.cpu()
    R = R.cpu()

    n = A_set.shape[0]
    bs = 100
    loss = 0
    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i*bs:min((i+1)*bs, n)]
        # SA = torch.matmul(S, AM)
        # AR = torch.matmul(AM, R)
        #
        # U1, Sig1, V1 = torch.svd(AR)
        # AR_pinv = V1.matmul(torch.diag_embed(1.0 / Sig1)).matmul(U1.permute(0, 2, 1))
        # U2, Sig2, V2 = torch.svd(SA)
        # SA_pinv = V2.matmul(torch.diag_embed(1.0 / Sig2)).matmul(U2.permute(0, 2, 1))
        #
        # prod = AR_pinv.matmul(AM).matmul(SA_pinv)
        # U3, Sig3, V3 = torch.svd(prod)
        # X = U3[:, :, :k].matmul(torch.diag_embed(Sig3[:, :k])).matmul(V3.permute(0, 2, 1)[:, :k])
        # ans = AR.matmul(X).matmul(SA)
        # it_loss = torch.sum(torch.norm(ans - AM, dim=(1, 2)))/n
        # loss += it_loss.item()
        SA = torch.matmul(S, AM)
        AR = torch.matmul(AM, R)
        SAR = S.matmul(AR)
        U1, Sig1, V1 = torch.svd(SAR)
        # prod = AR.matmul(V1).matmul(V1.permute(0, 2, 1))
        U2, Sig2, V2 = torch.svd(AR.matmul(V1))
        Y = U2[:, :, :k].matmul(torch.diag_embed(Sig2[:, :k])).matmul(V2.permute(0, 2, 1)[:, :k]).matmul(V1.permute(0,2,1))
        SAR_pinv = V1.matmul(torch.diag_embed(1.0 / Sig1)).matmul(U1.permute(0, 2, 1))
        ans = Y.matmul(SAR_pinv).matmul(SA)
        it_loss = torch.sum(torch.norm(ans - AM, dim=(1, 2)))/n
        loss += it_loss.item()
    return loss

def evaluate_to_rule_them_all_regression(A_set, B_set, S):
    """
    BATCHED, but also iterative (i.e. for data=hyper, eval list may be ~3000)
    :param A: list of matrices (3D tensor)
    :param sketch: S or [S, S2] concatenated; assumed matrices
    :param k: low-rank k
    :return: K-rk approx cost, averaged over matrices in eval_list
    """
    # A = eval_list
    n = A_set.size()[0]
    bs = 100
    loss = 0
    # print("ln 149 in eval: check everything on cpu")
    # IPython.embed()
    for i in range(math.ceil(n/float(bs))):
        AM = A_set[i*bs:min(n, (i+1)*bs)]
        BM = B_set[i*bs:min(n, (i+1)*bs)]

        SA = torch.matmul(S.cpu(), AM)
        SB = torch.matmul(S.cpu(), BM)
        U, Sig, V = torch.svd(SA)
        X = V.matmul(torch.diag_embed(1.0 / Sig)).matmul(U.permute(0, 2, 1)).matmul(SB)
        ans = AM.matmul(X)
        it_loss = torch.sum(torch.norm(ans - BM, dim=(1, 2)))/n
        loss += it_loss.item()
    # print("in evaluate.py, ln 154")
    # IPython.embed()
    return loss

def evaluate_to_rule_them_all(eval_list, sketch, k):
    """
    BATCHED, but also iterative (i.e. for data=hyper, eval list may be ~3000)
    :param A: list of matrices (3D tensor)
    :param sketch: S or [S, S2] concatenated; assumed matrices
    :param k: low-rank k
    :return: K-rk approx cost, averaged over matrices in eval_list
    """
    # A = eval_list
    n = eval_list.size()[0]
    bs = 100
    loss = 0
    for i in range(math.ceil(n/float(bs))):
        AM = eval_list[i*bs:min(n, (i+1)*bs)]
        SA = torch.matmul(sketch.cpu(), AM)
        U2, Sigma2, V2 = torch.svd(SA)
        AU = AM.matmul(V2)
        U3, Sigma3, V3 = torch.svd(AU)
        ans = U3[:, :, :k].matmul(torch.diag_embed(Sigma3[:, :k])).matmul(V3.permute(0, 2, 1)[:, :k]).matmul(
            V2.permute(0, 2, 1))
        it_loss = torch.sum(torch.norm(ans - AM, dim=(1, 2)))/n
        loss += it_loss.item()
    # print("in evaluate.py, ln 154")
    # IPython.embed()
    return loss

def evaluate_to_rule_them_all_sparse(eval_list, sketch, k):
    """
    Not batched; uses GPU within iteration
    :param eval_list:
    :param sketch:
    :param k:
    :param device:
    :return:
    """
    device = sketch.device.type + (":%d" % sketch.device.index if sketch.device.index else "")
    loss = 0
    n = len(eval_list)
    for A in eval_list:
        AM = A['M'][None].to(device)
        # Ad = A['d']
        # An = A['n']
        AMap = A['Map']

        # IPython.embed()
        ind = torch.tensor(AMap).type(torch.LongTensor).to(device)
        S = torch.index_select(sketch, dim=1, index=ind)
        SA = S.matmul(AM)
        U2, Sigma2, V2 = torch.svd(SA)
        AU = AM.matmul(V2)
        U3, Sigma3, V3 = torch.svd(AU)
        ans = U3[:, :, :k].matmul(torch.diag_embed(Sigma3[:, :k])).matmul(V3.permute(0, 2, 1)[:, :k]).matmul(
            V2.permute(0, 2, 1))
        # IPython.embed()
        loss += torch.norm(ans - AM, dim=(1, 2)).item()/n
    return loss

def evaluate_extra(sparse, eval_list,sketch_vector, sketch_value,sketch_vector2, sketch_value2,m,mextra,k,n,d):
    totLoss = 0
    count = 0
    for A in eval_list:
        if sparse:
            AM=A['M'].to(device)
            SA = torch.Tensor(m+mextra, A['d']).fill_(0).to(device)
            for i in range(A['n']):  # A has this many rows, not mapped yet
                actR = A['Map'][i]  # Actual row in the matrix
                mapR = sketch_vector[actR]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[actR]  # remember: times the weight

                mapR=sketch_vector2[actR]+m
                SA[mapR]+= AM[i] * sketch_value2[actR]
        else:
            AM=A.to(device)
            SA = torch.Tensor(m+mextra, d).fill_(0).to(device)
            for i in range(n):  # A has this many rows, not mapped yet
                mapR = sketch_vector[i]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight

                mapR=sketch_vector2[i]+m
                SA[mapR] += AM[i] * sketch_value2[i]  # remember: times the weight

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
        # if (count % 10 == 0):
        #     print(count, end=",")
        #     sys.stdout.flush()
    return totLoss

def getAvgDim(A_list):
    nL=[]
    dL=[]
    for A in A_list:
        nL.append(A['n'])
        dL.append(A['d'])
    print('Avg height',np.average(nL),'Avg width',np.average(dL))

def getbest(A_train, A_test,k,data,best_file):
    best_train = bestPossible(A_train, k, data).tolist()
    best_test = bestPossible(A_test,k,data).tolist()
    best_errs = [best_train/len(A_train) if len(A_train) != 0 else 0, best_test/len(A_test) if len(A_test) !=0 else 0]
    print(best_errs)
    torch.save(best_errs, best_file)
    return best_train, best_test

def bestPossible_regression(A_set, B_set):
    n = A_set.size()[0]
    bs = 100
    loss = 0
    # print("ln 278 in eval: check everything on cpu")
    # IPython.embed()
    for i in range(math.ceil(n/float(bs))):
        AM = A_set[i*bs:min(n, (i+1)*bs)]
        BM = B_set[i*bs:min(n, (i+1)*bs)]

        U, Sig, V = torch.svd(AM)
        X = V.matmul(torch.diag_embed(1.0 / Sig)).matmul(U.permute(0, 2, 1)).matmul(BM)
        ans = AM.matmul(X)
        # IPython.embed()
        it_loss = torch.sum(torch.norm(ans - BM, dim=(1, 2)))/n
        loss += it_loss.item()
    return loss

def getbest_regression(A_train, B_train, A_test, B_test, best_file):
    best_train_err = bestPossible_regression(A_train, B_train)
    best_test_err = bestPossible_regression(A_test, B_test)

    torch.save([best_train_err, best_test_err], best_file)
    return best_train_err, best_test_err