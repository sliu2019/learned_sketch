import numpy as np 
import torch
import sys
import IPython
import os
import pickle
from evaluate import evaluate_to_rule_them_all, evaluate_to_rule_them_all_sparse, evaluate_to_rule_them_all_regression, evaluate_to_rule_them_all_rsketch, evaluate_to_rule_them_all_4sketch
import warnings
import matplotlib.pyplot as plt
from global_variables import *
import math
import re
from sklearn.cluster import DBSCAN
import numpy_indexed as npi
from collections import Counter
from sparsity_pattern_init_algs import *

def return_data_fldr_pth(fldr_nm):
	hostname = get_hostname()
	if "ourlab" in hostname:
		data_fldr_pth = "/home/me/%s" % fldr_nm
	else:
		data_fldr_pth = "/Users/me/research/%s" % fldr_nm
	return data_fldr_pth

def IRLS(yM, XM, maxiter, w_init=1, d=0.0001, tolerance=0.001):
    nsamples, nx, ny = XM.shape
    #X = XM.reshape((nsamples*nx, ny)).to(device)
    X = XM.reshape((nsamples*nx, ny)).cpu()
    del XM
    nsamples, nx, ny = yM.shape
    #y = yM.reshape((nsamples*nx, ny)).to(device)
    y = yM.reshape((nsamples*nx, ny)).cpu()
    del yM

    n, p = X.shape
    delta_ = np.array(np.repeat(d, n)).reshape(1, n)
    delta = torch.from_numpy(delta_).cpu()
    w = np.repeat(1, n)
    #W = torch.from_numpy(np.diag(w)).to(device)
    W = torch.from_numpy(np.diag(w)).cpu()


    temp1 = torch.matmul(torch.t(X), W.float())
    para1 = torch.inverse(torch.matmul(temp1, X))
    temp2 = torch.matmul(torch.t(X), W.float())
    para2 = torch.matmul(temp2, y)
    #B = torch.matmul(para1, para2).to(device)
    B = torch.matmul(para1, para2).cpu()

    for _ in range(maxiter):
        #B_add = torch.zeros(B.shape[0]).to(device)
        B_add = torch.zeros(B.shape[0], B.shape[1]).cpu()

        _B = B + B_add
        #_w = torch.t(abs(y - torch.matmul(X, B.float()))).to(device)
        _w = torch.t(abs(y - torch.matmul(X, B.float()))).cpu()

        _w = _w.to(torch.float32)
        #delta = delta.to(torch.float32).to(device)
        delta = delta.to(torch.float32).cpu()

        w = float(1)/torch.max(delta, _w)
        W = torch.diag(w[0])
        temp1 = torch.matmul(torch.t(X).double(), W.double())
        para1 = torch.inverse(torch.matmul(temp1, X.double()))
        temp2 = torch.matmul(torch.t(X).double(), W.double())
        para2 = torch.matmul(temp2, y.double())
        B = torch.matmul(para1, para2)
        crit = torch.nn.SmoothL1Loss()
        loss = crit(_B.float(), B.float())
        tol = torch.sum(loss)
        # print("Tolerance = %s" % tol)
        if tol < tolerance:
            return B
    return B

def huber_regression(A, B):
    X = IRLS(yM=B, XM=A, maxiter=50)
    return X

def lp_norm_regression(p, A, B):
    X = IRLS(lp=p, yM=B, XM=A, maxiter=50)
    return X

def mysvd(init_A,k):
    if k>min(init_A.size(0),init_A.size(1)):
        k=min(init_A.size(0),init_A.size(1))
    d=init_A.size(1)
    x=[torch.Tensor(d).uniform_() for i in range(k)]
    for i in range(k):
        x[i]=x[i].to(device)
        x[i].requires_grad=False
    def perStep(x,A):
        x2=A.t().mv(A.mv(x))
        x3=x2.div(torch.norm(x2))
        return x3
    U=[]
    S=[]
    V=[]
    Alist=[init_A]
    for kstep in range(k): #pick top k eigenvalues
        cur_list=[x[kstep]]   #current history
        for j in range(300):  #steps
            cur_list.append(perStep(cur_list[-1],Alist[-1]))  #works on cur_list
        V.append((cur_list[-1]/torch.norm(cur_list[-1])).view(1,cur_list[-1].size(0)))
        S.append((torch.norm(Alist[-1].mv(V[-1].view(-1)))).view(1))
        U.append((Alist[-1].mv(V[-1].view(-1))/S[-1]).view(1,Alist[-1].size(0)))
        Alist.append(Alist[-1]-torch.ger(Alist[-1].mv(cur_list[-1]), cur_list[-1]))
    return torch.cat(U,0).t(),torch.cat(S,0),torch.cat(V,0).t()

def get_hostname():
	# IPython.embed()
	if sys.platform == "darwin":
		return "GS19624" # me's laptop
	else:
		with open("/etc/hostname") as f:
			hostname=f.read()
		hostname=hostname.split('\n')[0]
		print(hostname)
		return hostname

def args_to_fldrname(runtype, args):
	"""
	:param args: from parse_args(), a namespace
	:return: str, foldername
	"""
	ignore_keys = ["save_fldr","save_file", "single", "bestonly", "bw", "dense", "dwnsmp", "lr_S", "raw", "load_file", "device", "lev_count", "lev_cutoff", "lev_ridge"]
	d_args = vars(args)
	fldrname = runtype
	for key in sorted(d_args.keys()):
		if key not in ignore_keys:
			# print(key, d_args[key])
			fldrname += "_" + str(key) + "_" + str(d_args[key])
	# IPython.embed()
	return fldrname

def save_iteration_4sketch(S, R, T, W, A_train, A_test, args, save_dir, bigstep):
	torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)

	test_err = evaluate_to_rule_them_all_4sketch(A_test, S, R, T, W, args.k)
	train_err = 0
	# train_err = evaluate_to_rule_them_all_4sketch(A_train, S, R, T, W, args.k)
	torch.save([[S, R, T, W], [train_err, test_err]], torch_save_fpath)

	print(train_err, test_err)
	print("Saved iteration: %d" % bigstep)
	return train_err, test_err

def save_iteration_rsketch(S, R, A_train, A_test, args, save_dir, bigstep):
	torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)

	test_err = evaluate_to_rule_them_all_rsketch(A_test, S, R, args.k)
	train_err = 0
	# train_err = evaluate_to_rule_them_all_rsketch(A_train, S, R, args.k)
	torch.save([[S, R], [train_err, test_err]], torch_save_fpath)

	print(train_err, test_err)
	print("Saved iteration: %d" % bigstep)
	return train_err, test_err

def save_iteration_regression(S, A_train, B_train, A_test, B_test, save_dir, bigstep):
	"""
	Not implemented:
	Mixed matrix evaluation
	"""
	torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)

	test_err = evaluate_to_rule_them_all_regression(A_test, B_test, S)
	# train_err = 0
	train_err = evaluate_to_rule_them_all_regression(A_train, B_train, S)
	torch.save([[S], [train_err, test_err]], torch_save_fpath)

	print(train_err, test_err)
	print("Saved iteration: %d" % bigstep)
	return train_err, test_err

def save_iteration(S, A_train, A_test, args, save_dir, bigstep, type=None, S2=None, sparse=False):
	if sparse:
		eval_fn = evaluate_to_rule_them_all_sparse
	else:
		eval_fn = evaluate_to_rule_them_all

	warnings.warn("Save iteration does not handle 'tech' or sparse type data")
	if type is None:
		torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)
	else:
		torch_save_fpath = os.path.join(save_dir, str(type) + "_it_%d" % bigstep)

	if S2 is None:
		test_err = eval_fn(A_test, S, args.k)
		train_err = 0
		# train_err = eval_fn(A_train, S, args.k)
		torch.save([[S], [train_err, test_err]], torch_save_fpath)
	else:
		test_err = eval_fn(A_test, torch.cat([S, S2]), args.k)
		train_err = eval_fn(A_train, torch.cat([S, S2]), args.k)
		torch.save([[S, S2], [train_err, test_err]], torch_save_fpath)

	print(train_err, test_err)
	print("Saved iteration: %d" % bigstep)
	return train_err, test_err

######## KMEANS ########
def initialize_centroids(k, points):
	"""returns k centroids from the initial points"""
	centroids = points.copy()
	np.random.shuffle(centroids)
	return centroids[:k]

def closest_centroid(points, centroids):
	"""returns an array containing the index to the nearest centroid for each point"""
	distances = np.linalg.norm(points - centroids[:, np.newaxis], axis=2)
	# IPython.embed()
	return np.argmin(distances, axis=0)

def update_centroids(points, closest, centroids):
	"""returns the new centroids assigned from the points closest to them"""
	# IPython.embed()
	new_centroids = np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
	return new_centroids

def run_kmeans(data, k_means):
	centroids = initialize_centroids(k_means, data)

	dist = float("inf")
	count = 0
	while dist > 0.05 and count<20: # stopping condition
		closest = closest_centroid(data, centroids)
		new_centroids = update_centroids(data, closest, centroids)
		dist = np.linalg.norm(new_centroids - centroids)
		print(dist)
		centroids = new_centroids
		count +=1
	return centroids

def init_w_kmeans(A_train, m, rk_k):
	# TODO: how to use EACH matrix in A_train?
	"""
	Note: currently only uses the first matrix in set A_train
	"""
	rand_ind = np.random.randint(low=0, high=len(A_train))
	print("sampled matrix %d" % rand_ind)
	A_train_sample = A_train[rand_ind].numpy()
	A_train_sample = (A_train_sample.T/np.linalg.norm(A_train_sample, axis=1)).T
	centroids = run_kmeans(np.copy(A_train_sample), m)
	rv = closest_centroid(np.copy(A_train_sample), np.copy(centroids))
	# visualize_kmeans(A_train_sample, centroids)
	rv = torch.from_numpy(rv)
	# A_train = A_train.numpy()

	return rv

def visualize_kmeans(A_train_sample, centroids):
	u, s, vt = np.linalg.svd(A_train_sample)
	proj_sample = A_train_sample@(vt[:2].T)
	proj_centroids = centroids@(vt[:2].T)

	plt.scatter(proj_sample[:, 0], proj_sample[:, 1])
	for i in range(proj_centroids.shape[0]):
		plt.plot([0, proj_centroids[i, 0]], [0, proj_centroids[i, 1]])
	plt.savefig("visualize_kmeans.jpg")

def init_w_lev_S(A_train, m, rk_k, ridge, cutoff=None, count=False):
	"""
	Same as init_w_lev, but computes an S for every A
	"""
	k = rk_k
	if not cutoff:  # arg is None
		cutoff = m // 2  # [m - 5, int(m/2.0), m//2]

	tr_len = A_train.size()[0]
	d = A_train.size()[2]
	n = A_train.size()[1]
	N_train = A_train.size()[0]

	tot_lev_scores = torch.zeros(n)
	all_lev_scores = torch.empty(0, n)  # debug
	for i in range(math.ceil(tr_len / 50)):
		print("it %d of init_w_lev_S" % i)
		A = A_train[i * 50: min((i + 1) * 50, tr_len)]
		bs = A.size()[0]

		# Compute lambdas
		if ridge:
			u, s, v = torch.svd(A)
			A_k = u[:, :, :k].matmul(torch.diag_embed(s[:, :k])).matmul(v.permute(0, 2, 1)[:, :k])
			lambd_old = torch.norm(A_k - A, dim=(1, 2))
			lambd = (lambd_old ** 2) / k
		else:
			lambd = torch.zeros(bs)

		x = torch.matmul(A.permute(0, 2, 1), A) + torch.mul(lambd[:, None, None],
															(torch.eye(d)[None, :, :]).repeat(bs, 1, 1))
		inv = torch.inverse(x)
		rr_lev_scores = torch.diagonal(A.matmul(inv).matmul(A.permute(0, 2, 1)), dim1=1, dim2=2)
		if i == 0:
			all_lev_scores = all_lev_scores.type(rr_lev_scores.dtype)
		all_lev_scores = torch.cat((all_lev_scores, rr_lev_scores))
		tot_lev_scores += torch.sum(rr_lev_scores, dim=0)

	topk_ind = torch.topk(all_lev_scores, cutoff, dim=1)[1].type(torch.LongTensor)
	# print("ln 184 in misc_utils.py")
	# IPython.embed() # should be size (N_train, k) and has integer contents

	if cutoff < m:
		sketch_vector = torch.randint(cutoff, m, [N_train, n])
		sketch_vector[torch.arange(N_train).repeat_interleave(cutoff), topk_ind.reshape(-1)] = torch.arange(cutoff).repeat(N_train)
	elif cutoff == m:
		sketch_vector = torch.randint(m, [N_train, n])
		sketch_vector[torch.arange(N_train).repeat_interleave(cutoff), topk_ind.reshape(-1)] = torch.arange(
			cutoff).repeat(N_train)
	else:
		raise Exception("cutoff > m for leverage score initialization")

	del u, s, v, A_k, lambd, lambd_old, x, inv, rr_lev_scores, all_lev_scores, tot_lev_scores
	torch.cuda.empty_cache()
	return sketch_vector

def init_w_lev(A_train, m, rk_k, ridge, cutoff=None, count=False):
	"""
	Computes m/2 top row ridge leverage scores and places in own bucket
	Random hash for rest of rows
	Batched
	:param A_train: 3D tensor
	:param m: S is m x n
	:param rk_k: rank k of the problem at large
	:return:
	"""
	# print("in init_w_lev")
	# IPython.embed()
	k = rk_k
	# TODO: UPDATE VALUE
	if not cutoff: # arg is None
		cutoff = m//2 #[m - 5, int(m/2.0), m//2]

	tr_len = A_train.size()[0]
	d = A_train.size()[2]
	n = A_train.size()[1]

	tot_lev_scores = torch.zeros(n)
	all_lev_scores = torch.empty(0, n) # debug
	for i in range(math.ceil(tr_len / 50)):
		A = A_train[i*50: min((i+1)*50, tr_len)]
		bs = A.size()[0]

		# Compute lambdas
		if ridge:
			u, s, v = torch.svd(A)
			A_k = u[:, :, :k].matmul(torch.diag_embed(s[:, :k])).matmul(v.permute(0, 2, 1)[:, :k])
			lambd_old = torch.norm(A_k - A, dim=(1, 2))
			lambd = (lambd_old**2)/k
			# print(lambd_old)
			# print(lambd)
		else:
			lambd = torch.zeros(bs)

		x = torch.matmul(A.permute(0, 2, 1), A) + torch.mul(lambd[:,None, None], (torch.eye(d)[None, :,:]).repeat(bs, 1, 1))
		inv = torch.inverse(x)
		rr_lev_scores = torch.diagonal(A.matmul(inv).matmul(A.permute(0, 2, 1)), dim1=1, dim2=2)
		if i == 0:
			all_lev_scores = all_lev_scores.type(rr_lev_scores.dtype)
		all_lev_scores = torch.cat((all_lev_scores, rr_lev_scores))
		tot_lev_scores += torch.sum(rr_lev_scores, dim=0)

	# print("VALUE OF LARGEST: ", largest)
	if count:
		topk_ind_all = torch.topk(all_lev_scores, cutoff, dim=1)[1]
		topk_ind_all_counter = Counter(list(topk_ind_all.numpy().flatten()))
		popular_ind = sorted(topk_ind_all_counter, key=topk_ind_all_counter.get, reverse=True)
		# print("count method, all unioned: ", topk_ind_all_counter.most_common(k))
		# topk_ind_count = set(popular_ind[:k])
		topk_ind = torch.from_numpy(np.array(popular_ind[:k])).type(torch.LongTensor)
		# IPython.embed()
	else:
		topk_ind = torch.topk(tot_lev_scores, cutoff)[1].type(torch.LongTensor)
	# print(np.sort(topk_ind.numpy())) # TODO

	if cutoff < m:
		sketch_vector = torch.randint(cutoff, m, [n]).int()
		sketch_vector[topk_ind] = torch.arange(cutoff).int()
	elif cutoff == m:
		sketch_vector = torch.randint(m, [n]).int()
		sketch_vector[topk_ind] = torch.arange(cutoff).int()
	else:
		raise Exception("cutoff > m for leverage score initialization")

	# TODO: debug: avg = 2.689054726368159 on size=500 video, friends data
	"""old_set = set(topk_ind.numpy())
	avg = 0
	for j in range(tr_len):
		topk = torch.topk(all_lev_scores[j], cutoff)[1]
		new_set = set(list(topk.numpy()))
		int = len(new_set.intersection(old_set))
		avg += int
		# print(int)
		# old_set = new_set
	# print(avg/tr_len)"""

	return sketch_vector #TODO Debug, all_lev_scores, topk_ind

def init_w_lev_cluster(A_train, m, rk_k, ridge, cutoff):
	"""
	Computes m/2 top row ridge leverage scores and places in own bucket
	Random hash for rest of rows
	Batched
	:param A_train: 3D tensor
	:param m: S is m x n
	:param rk_k: rank k of the problem at large
	:return:
	"""
	k = rk_k
	# TODO: UPDATE VALUE
	if not cutoff: # arg is None
		cutoff = m//2 #[m - 5, int(m/2.0), m//2]

	tr_len = A_train.size()[0]
	d = A_train.size()[2]
	n = A_train.size()[1]

	tot_lev_scores = torch.zeros(n)
	all_lev_scores = torch.empty(0, n) # debug
	for i in range(math.ceil(tr_len / 50)):
		A = A_train[i*50: min((i+1)*50, tr_len)]
		bs = A.size()[0]

		# Compute lambdas
		if ridge:
			u, s, v = torch.svd(A)
			A_k = u[:, :, :k].matmul(torch.diag_embed(s[:, :k])).matmul(v.permute(0, 2, 1)[:, :k])
			lambd_old = torch.norm(A_k - A, dim=(1, 2))
			lambd = lambd_old**2/k
			# print(lambd_old)
			# print(lambd)
			# lambd < lambd_old/2
		else:
			lambd = torch.zeros(bs)

		x = torch.matmul(A.permute(0, 2, 1), A) + torch.mul(lambd[:,None, None], (torch.eye(d)[None, :,:]).repeat(bs, 1, 1))
		inv = torch.inverse(x)
		rr_lev_scores = torch.diagonal(A.matmul(inv).matmul(A.permute(0, 2, 1)), dim1=1, dim2=2)
		all_lev_scores = torch.cat((all_lev_scores, rr_lev_scores))
		tot_lev_scores += torch.sum(rr_lev_scores, dim=0)

	topk_ind = cluster(A_train, all_lev_scores, cutoff, m)

	if cutoff < m:
		sketch_vector = torch.randint(cutoff, m, [n]).int()
		sketch_vector[topk_ind] = torch.arange(cutoff).int()
	elif cutoff == m:
		sketch_vector = torch.randint(m, [n]).int()
		sketch_vector[topk_ind] = torch.arange(cutoff).int()
	else:
		raise Exception("cutoff > m for leverage score initialization")

	return sketch_vector

def cluster(A, all_lev_scores, n_cutoff, m):
	"""
	For each matrix in A:
	Clusters top (largest lev score) n_to_cluster rows
	From the top n_cutoff clusters, choose a core sample
	Add index of core sample to list
	"""
	bs = A.size()[0]
	n = A.size()[1]
	n_to_cluster = m*10

	# IPython.embed()
	subset1_lev_scores, subset1_ind = torch.topk(all_lev_scores, n_to_cluster, dim=1)
	A_subset1 = torch.reshape(A[torch.repeat_interleave(torch.arange(bs), repeats=n_to_cluster), torch.flatten(subset1_ind)], (bs, n_to_cluster, -1))
	final_inds = np.zeros((bs, n_cutoff)).astype("int64")
	for i in range(bs):
		A_subset1i = A_subset1[i]
		db = DBSCAN(eps=1, min_samples=2).fit(A_subset1i) # 0.08
		labels = db.labels_

		# IPython.embed()
		print(len(np.unique(labels)))
		subset1i_lev_scores = subset1_lev_scores[i]
		subset1i_ind = subset1_ind[i]
		# For each cluster, compute the avg lev score and the representative datapoint
		a = np.concatenate((labels[None, :], subset1i_lev_scores[None, :]), axis=0)
		lev_scores_clustered = npi.group_by(a[0]).split(a[1])
		lev_scores_clustered.pop(0)
		avg_lev_by_cluster = np.array([np.mean(x) for x in lev_scores_clustered])

		# IPython.embed()
		# pick a random cluster member as representative
		# TODO: can select closest member to mean instead
		# TODO: select any among the core points
		# Wait: is every cluster guaranteed to have a core point? Yes
		a = np.concatenate((labels[None, db.core_sample_indices_], np.arange(len(labels))[None, db.core_sample_indices_]),axis=0)
		subset1i_indices_clustered = npi.group_by(a[0]).split(a[1])
		# subset1i_indices_clustered.pop(0)
		rep_ind_by_cluster = np.array([np.random.choice(x) for x in subset1i_indices_clustered])

		# IPython.embed()
		# Find topk on clustered lev scores
		subset2i_clustered_ind = np.where(labels != -1)
		inds = np.delete(np.arange(n), subset1i_ind[subset2i_clustered_ind])
		inds = np.append(inds, subset1i_ind[rep_ind_by_cluster])
		lev_scores = np.delete(all_lev_scores[i], subset1i_ind[subset2i_clustered_ind])
		lev_scores = np.append(lev_scores, avg_lev_by_cluster)

		# IPython.embed()
		top_inds = inds[np.argsort(lev_scores)[-n_cutoff:]]
		final_inds[i] = top_inds
	return final_inds, subset1i_ind, labels

def find_P(A, k):
	"""
	:param A: bs x n x d
	:return:
	"""
	A = np.multiply(A, 1.0/np.linalg.norm(A, axis=2)[:,:,None]) # normalize rows??? yes or no?
	bs = A.shape[0]
	d = A.shape[2]
	n = A.shape[1]

	for i in range(k):
		print(i)
		if i == 0:
			# IPython.embed()
			# TODO: initialize all differently or the same?
			# inds = np.random.randint(0, n, size=(bs, 1)) # bs x i
			inds = np.random.randint(0, n)*np.ones((bs, 1)).astype("int")
			# IPython.embed()
			# TODO: does this select rows?
			P = A[np.arange(bs), inds[:,0]][:, None, :] # bs x i x d
		else:
			# IPython.embed()
			U, S, VT = np.linalg.svd(P)
			V_s = np.swapaxes(VT[:, :i], 1, 2)
			proj = np.matmul(np.matmul(A, V_s), np.swapaxes(V_s, 1, 2))
			scores = np.linalg.norm(A - proj, axis=2)
			ind = np.argmax(scores, axis=1)
			new_p = A[np.arange(bs), ind][:, None, :]
			P = np.concatenate((P, new_p), axis=1)
			inds = np.concatenate((inds, ind[:, None]), axis=1)
			assert np.unique(inds, axis=1).shape == inds.shape
	return inds

def init_w_gramschmidt(A_train, m, rk_k, size, bestfile_save_dir):
	"""
	USES NUMPY
	:param A_train:
	:param m:
	:param rk_k:
	:return:
	"""
	k = rk_k
	tr_len = A_train.size()[0]
	d = A_train.size()[2]
	n = A_train.size()[1]
	cutoff = m-5 # TODO: don't forget to set this!!
	A_train = A_train.numpy()

	save_fl = os.path.join(bestfile_save_dir, "gs_N_%d_top_%d.npy" % (size, cutoff))
	fls = [x for x in os.listdir(bestfile_save_dir) if
		   os.path.splitext(x)[1] == ".npy" and int(re.search(r"N_(.*)_top", x).group(1)) == size]
	top_values = [int(re.search(r"top_(.*).npy", x).group(1)) for x in fls]
	for top_value in top_values:
		if top_value > cutoff:
			inds = np.load(os.path.join(bestfile_save_dir, "gs_N_%d_top_%d.npy" % (size, cutoff)))
			break
	else:
		inds = find_P(A_train, cutoff)
		np.save(save_fl, inds)
	# IPython.embed()
	unique, unique_counts = np.unique(inds, return_counts=True)
	sorted_ind = np.argsort(unique_counts)
	topk = unique[sorted_ind[-cutoff:]]
	sketch_vector = np.random.randint(cutoff, m, size=n)
	sketch_vector[topk] = np.arange(cutoff)
	# IPython.embed() # check types
	return torch.from_numpy(sketch_vector)

def init_w_load(load_file, exp_num, n, m):
	"""
	CAUTION: Should only be used for greedy experiments
	:param load_file:
	:param exp_num:
	:return:
	"""
	full_flpth = os.path.join("/home/me/research/big-lowrank", load_file, "exp_%d" % exp_num, "saved_tensors_it_%d"%(int(n//2)-m-1))
	if not os.path.exists(full_flpth):
		print("CAUTION: NOT LOADING THE ONE YOU WERE EXPECTING")
		it_num = int((n//2)-m-1)
		it_num = (it_num//200)*200
		full_flpth = os.path.join("/home/me/research/big-lowrank", load_file, "exp_%d" % exp_num, "saved_tensors_it_%d"%it_num)
	print("LOADING SKETCH PATTERN AND VALUES FROM %s" % full_flpth)
	x = torch.load(full_flpth)
	# IPython.embed()

	sketch_vector = x[0]
	sketch_value = x[1]
	active_ind = x[2]
	return sketch_vector, sketch_value, active_ind


