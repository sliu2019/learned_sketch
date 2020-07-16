import numpy as np, torch, IPython, os
from data.videos import *
from global_variables import *
from tqdm import tqdm
# import tensorflow as tf
import time
import argparse
from data.hyperspectra import getHyper
from data.tech import getTech

def compute_proj_loss(A, sketch_vector, sketch_value, m):
	n = A.size()[1]

	S = torch.zeros((m, n))
	S[sketch_vector, torch.arange(n)] = sketch_value
	SA = S.matmul(A)
	U, Sig, V = torch.svd(SA)

	proj = A.matmul(V).matmul(V.permute(0, 2, 1))
	loss = torch.mean(torch.norm(A - proj, dim=(1, 2)))
	return loss

def compute_full_loss(A, sketch_vector, sketch_value, m, k):
	n = A.size()[1]

	S = torch.zeros((m, n))
	S[sketch_vector, torch.arange(n)] = sketch_value
	SA = S.matmul(A)
	U, Sig, V = torch.svd(SA)

	AU = A.matmul(V)
	U3, Sigma3, V3 = torch.svd(AU)
	ans = U3[:, :, :k].matmul(torch.diag_embed(Sigma3[:, :k])).matmul(
		V3.permute(0, 2, 1)[:, :k]).matmul(V.permute(0, 2, 1))
	loss = torch.mean(torch.norm(ans - A, dim=(1, 2)))
	return loss

def evaluate(A_train, sketch_vector, sketch_value, m, k):
	N_train = A_train.size()[0]
	n = A_train.size()[1]
	d = A_train.size()[2]

	full_loss, proj_loss = 0, 0
	for i in range(math.ceil(N_train/50)):
		ind_2 = min(N_train, (i+1)*50)
		A_section = A_train[i*50: ind_2]

		full_loss += compute_full_loss(A_section, sketch_vector, sketch_value, m, k)*((ind_2 - i*50)/N_train)
		proj_loss += compute_proj_loss(A_section, sketch_vector, sketch_value, m)*((ind_2 - i*50)/N_train)

	return proj_loss, full_loss

def update_sketch_values(A_train, A_test, sketch_vector, old_sketch_value, m, k, active_ind, device, LR=10, num_its=1000):
	"""
	Assumptions:
	Proj loss (rather than full loss)

	:param A_train:
	:param A_test:
	:param sketch_vector: cpu
	:param sketch_values: cpu
	:param m:
	:param k:
	:param active_ind:
	:param LR:
	:param num_its:
	:return:
	"""
	N_train = A_train.size()[0]
	n = A_train.size()[1]
	d = A_train.size()[2]

	sketch_value = old_sketch_value.data
	sketch_value.requires_grad = True

	print_freq = 200
	bs = 5
	print("Retraining sketch_values")
	for i in range(num_its):
		if (i % print_freq) == 0:
			print("it %d" % i)
			# proj_loss, full_loss = evaluate(A_train, sketch_vector, sketch_value, m, k)
			# print("Train: %f, %f" % (proj_loss, full_loss))
			# proj_loss, full_loss = evaluate(A_test, sketch_vector, sketch_value, m, k)
			# print("Test: %f, %f" % (proj_loss, full_loss))

		S = torch.zeros((m, n)).to(device)
		S[sketch_vector, torch.arange(n)] = sketch_value.to(device)
		AM = A_train[np.random.randint(0, N_train, bs)].to(device)
		SA = S.matmul(AM)
		U, Sig, V = torch.svd(SA)

		proj = AM.matmul(V).matmul(V.permute(0, 2, 1))
		loss = torch.mean(torch.norm(AM - proj, dim=(1, 2)))
		loss.backward()
		with torch.no_grad():
			sketch_value[active_ind] -= (LR/bs)*sketch_value.grad[active_ind]
			sketch_value.grad.zero_()
		del S, AM, SA, U, Sig, V, proj, loss
		torch.cuda.empty_cache()

	return sketch_value.data

def fast_loss(gs_samples, AM, U0, Sig0, V0, m, n, d, k, use_proj_loss, i, device, num_bins_sample, sampled_bins=None):
	"""
	:param gs_samples:
	:param AM: n x d
	:param U0: m x m
	:param Sig0: m x m
	:param V0: d x m
	:param m:
	:param n:
	:param d:
	:param num_bins_sample:
	:return:
	"""
	num_gs_samples = gs_samples.size()[0]
	if num_bins_sample == 0:
		num_bins_sample = m
		sampled_bins = np.arange(m)
	if num_bins_sample and sampled_bins is None:
		sampled_bins = np.random.choice(np.arange(m), size=num_bins_sample, replace=False)
	a = torch.zeros((num_bins_sample, m))
	a[np.arange(num_bins_sample), sampled_bins] = 1.0
	a = a[:, :, None]
	a = torch.repeat_interleave(a, num_gs_samples, dim=0)
	a = a.to(device)

	# (m*num_gs_samples) x d x 1
	b = (gs_samples[:, None].matmul(AM[i][None]))[:, :, None]
	b = b.repeat(num_bins_sample, 1, 1)
	b = b.to(device)

	V = fast_rank1_update_svd(U0, Sig0, V0, a, b, device)
	V = V[:, :, :m]
	del U0, Sig0, V0, a, b
	if use_proj_loss:
		total = num_bins_sample * num_gs_samples
		gpu_bs = 100
		proj_losses = torch.empty(total)
		# print("computing proj loss")
		for j in range(math.ceil(total / float(gpu_bs))):
			V_batch = V[j * gpu_bs: min(total, (j + 1) * gpu_bs)]
			proj = AM[None].matmul(V_batch).matmul(V_batch.permute(0, 2, 1))
			loss_tensor = torch.norm(AM[None] - proj, dim=(1, 2))
			proj_losses[j * gpu_bs: min(total, (j + 1) * gpu_bs)] = loss_tensor
			del V_batch, proj, loss_tensor
		losses = proj_losses
	else:
		# print("computing full loss")
		total = num_bins_sample * num_gs_samples
		gpu_bs = 50
		full_losses = torch.empty(total)
		for j in range(math.ceil(total / float(gpu_bs))):
			V_batch = V[j * gpu_bs: min(total, (j + 1) * gpu_bs)]
			AV = AM.matmul(V_batch)
			U3, Sigma3, V3 = torch.svd(AV)
			ans = U3[:, :, :k].matmul(torch.diag_embed(Sigma3[:, :k]).to(device)).matmul(
				V3.permute(0, 2, 1)[:, :k]).matmul(V_batch.permute(0, 2, 1))
			loss_tensor = torch.norm(ans - AM, dim=(1, 2))
			full_losses[j * gpu_bs: min(total, (j + 1) * gpu_bs)] = loss_tensor
			del V_batch, AV, U3, Sigma3, V3, ans, loss_tensor
		losses = full_losses
	return losses, sampled_bins

def args_to_fldrname_gs(args, parser):
	"""
	:param args: from parse_args(), a namespace
	:return: str, foldername
	"""
	ignore_keys = ["save_fldr","save_file","device", "data", "dataname"]
	d_args = vars(args)
	exp_fldr = args.save_file
	for key in sorted(d_args.keys()):
		if key not in ignore_keys and d_args[key] != parser.get_default(key):
			exp_fldr += "_" + str(key) + "_" + str(d_args[key])
	if not args.save_file:
		exp_fldr = exp_fldr[1:]
	exp_path = os.path.join("/home/me/research/big-lowrank/greedy", args.data, args.dataname if args.data=="video" else "", "gs", args.save_fldr, exp_fldr)
	return exp_path

def init_w_greedy_gs(A_train, A_test, args, parser, data, dataname, m, k, num_exp=1, num_A_sample=1, retrain_svalues_freq=0, num_gs_samples=10, use_proj_loss=True, n_early_factor=1, save_fldr="", device="cuda:0", LR=1.0, save_file="", switch_objectives=False, num_bins_sample=0, weird=False):
	print("Running exp on device %s" % device)
	N_train = A_train.size()[0]
	n = A_train.size()[1]
	d = A_train.size()[2]

	# early termination option
	end_ind = math.ceil(n*n_early_factor)

	exp_path = args_to_fldrname_gs(args, parser)
	for exp_num in range(num_exp):
		print("exp_num %d" % exp_num)
		if weird and (exp_num == 0 or exp_num == 1):
			continue
		save_path = os.path.join(exp_path, "exp_%d" % exp_num)

		if not os.path.exists(save_path):
			os.makedirs(save_path)

		sketch_vector = torch.zeros(n).type(torch.LongTensor)
		sketch_values = torch.zeros(n)

		shuff_row_ind = np.arange(n)
		np.random.shuffle(shuff_row_ind)
		m_init_ind = shuff_row_ind[:m]
		sketch_vector[m_init_ind] = torch.arange(m)
		sketch_values[m_init_ind] = torch.from_numpy(np.random.normal(size=m).astype("float32"))

		active_ind = m_init_ind

		bs = num_A_sample

		if num_bins_sample == 0:
			num_bins_sample = m

		AM = (A_train[np.random.randint(0, N_train, bs)]).to(device)

		count = 0
		print_freq = 50 # TODO

		# init save data structs
		test_errs = np.empty((0, 2))
		train_errs = np.empty((0, 2))
		exp_use_proj_loss = use_proj_loss
		for i in tqdm(shuff_row_ind[m:end_ind]):
			if count > 200:
				print_freq = 200
			if count == int((end_ind-m)//2) and switch_objectives:
				print("using full loss")
				exp_use_proj_loss = False

			gs_samples = torch.linspace(-2, 2, steps=num_gs_samples).to(device)
			S = torch.zeros((m, n)).to(device)
			S[sketch_vector, torch.arange(n)] = sketch_values.to(device)
			SA = S.matmul(AM)
			t0 = time.time()
			U0, Sig0, V0 = torch.svd(SA)

			avg_proj_losses = torch.zeros(num_bins_sample*num_gs_samples)
			sampled_bins = None
			for j in range(bs):
				j_proj_loss, sampled_bins = fast_loss(gs_samples, AM[j], U0[j], Sig0[j], V0[j], m, n, d, k, exp_use_proj_loss, i, device, num_bins_sample, sampled_bins)
				avg_proj_losses += j_proj_loss/float(bs)

			min_ind_flat = torch.argmin(avg_proj_losses)
			min_ind = [min_ind_flat//num_gs_samples, min_ind_flat % num_gs_samples]

			# update sketch vector/values
			sketch_vector[i] = torch.tensor(sampled_bins[min_ind[0]])
			sketch_values[i] = gs_samples[min_ind[1]]
			active_ind = np.concatenate((active_ind, [i]))

			if retrain_svalues_freq:
				if count % retrain_svalues_freq == 0:
					sketch_values = update_sketch_values(A_train, A_test, sketch_vector, sketch_values, m, k, active_ind, device, LR=LR)

			# every so often: evaluate (train and test) and save errors and sketch vector/values
			if count % print_freq == 0 or count == (end_ind-m-1):
				proj_loss, full_loss = evaluate(A_train, sketch_vector, sketch_values, m, k)
				train_errs = np.concatenate((train_errs, np.array([[proj_loss, full_loss]])), axis=0)
				print("it %d, train errs: %f, %f" % (count, proj_loss, full_loss))
				proj_loss, full_loss = evaluate(A_test, sketch_vector, sketch_values, m, k)
				test_errs = np.concatenate((test_errs, np.array([[proj_loss, full_loss]])), axis=0)
				print("it %d, test errs: %f, %f" % (count, proj_loss, full_loss))
				# IPython.embed()
				torch.save([sketch_vector, sketch_values, torch.from_numpy(active_ind)], os.path.join(save_path, "saved_tensors_it_%d" % count))
				np.save(os.path.join(save_path, "train_errs.npy"), train_errs)
				np.save(os.path.join(save_path, "test_errs.npy"), test_errs)

			count += 1

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
	m = V.size()[1]
	d = V.size()[0]

	m_tens = U[None].permute(0, 2, 1).matmul(a)
	p = a - U[None].matmul(m_tens) # a perp U
	R_a = torch.norm(p, dim=1)
	P = p *(1.0/R_a[:,:,None])

	n = V[None].permute(0, 2, 1).matmul(b)
	q = b - V[None].matmul(n)
	R_b = torch.norm(q, dim=1)
	Q = q *(1.0/R_b[:, :, None])

	S_ext = torch.zeros(m+1, m+1).to(device)
	S_ext[:m, :m] = torch.diag(Sig)
	y = torch.cat((n, R_b[:, :, None]), dim=1)
	K = S_ext + torch.cat((m_tens, R_a[:,:,None]), dim=1).matmul(y.permute(0, 2, 1))

	bs = a.size()[0]

	V_tiled = V[None].repeat(bs, 1, 1)
	V_ext = torch.cat((V_tiled, Q), dim=2)

	u1, s1, v1 = torch.svd(K)
	V_prime = V_ext.matmul(v1)  #anyways, s1[m] is tiny

	del m_tens, p, R_a, P, n, q, R_b, Q, S_ext, y, K, V_tiled, V_ext, u1, s1, v1
	return V_prime

def create_parser_gs():
	parser = argparse.ArgumentParser()

	def aa(*args, **kwargs):
		parser.add_argument(*args, **kwargs)

	aa("--data", type=str, default="video", help="tech|video|hyper")
	aa("--dataname", type=str, default="mit", help="eagle|mit|friends")
	# aa("--size", type=int, default=500, help="dataset size")

	aa("--m", type=int, default=None, help="m for S")
	aa("--k", type=int, default=None, help="target: rank k approximation")

	aa("--num_exp", type=int, default=1, help="number of trials for this experiment")
	aa("--retrain_svalues_freq", type=int, default=0, help="retrain s_value every x iterations?")
	aa("--num_gs_samples", default=10, type=int, help="number of samples in the range [-2, 2] for the row weights")
	aa("--use_full_loss", default=False, action="store_true", help="use full loss instead of proj loss")
	aa("--n_early_factor", type=float, default=1.0, help="Only place n*n_early_factor rows, instead of n")
	aa("--num_A_sample",type=int, default=1, help="Number of A matrices to sample")
	aa("--switch_objectives", default=False, action="store_true", help="use proj_loss for 1st half, full for 2nd")
	aa("--num_bins_sample", type=int, default=0, help="0 means all bins; else sample fewer")
	aa("--weird", default=False, action="store_true", help="resume exp 2, for a weird edge case")

	aa("--save_fldr", type=str, default="", help="describe what kind of exp this is")
	aa("--device", type=str, default="cuda:0", help="can set gpu per experiment")
	aa("--save_file", type=str, default="", help="describe what kind of exp this is")
	return parser

if __name__ == "__main__":
	parser = create_parser_gs()
	args = parser.parse_args()

	# IPython.embed()
	rawdir = "/home/me/research/big-lowrank/"
	if args.data == 'tech':
		print("Not implemented for tech")
		raise(NotImplementedError)
	elif args.data == 'hyper':
		raw = False
		A_train, A_test, n, d = getHyper(raw, 500, rawdir, 100)
		LR = 1.0
	elif args.data == 'video':
		raw = False
		A_train, A_test, n, d = getVideos(args.dataname, raw, 500, rawdir, 100, False, 1)
		LR = 10.0

	init_w_greedy_gs(A_train, A_test, args, parser, args.data, args.dataname, args.m, args.k, num_A_sample=args.num_A_sample, num_exp=args.num_exp, retrain_svalues_freq=args.retrain_svalues_freq, num_gs_samples=args.num_gs_samples, use_proj_loss=(not args.use_full_loss), n_early_factor=args.n_early_factor, save_fldr=args.save_fldr, device=args.device, LR=LR, save_file=args.save_file, switch_objectives=args.switch_objectives, num_bins_sample=args.num_bins_sample, weird=args.weird)
