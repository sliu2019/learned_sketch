import torch
import numpy as np
import GPUtil
import IPython

# Set device
if torch.cuda.is_available():
    # available_gpus = GPUtil.getAvailable(limit=2, maxLoad=0.5, maxMemory=0.009) # you can the edit params
    # GPUtil.showUtilization()
    # if len(available_gpus) > 0:
    #     device = torch.device("cuda:%d" % available_gpus[0])
    # else:
    #     device = torch.device("cuda:1")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# print("Running experiment on device: %s:%d" % (device.type, 0 if device.index is None else device.index))

# fix seeds for deterministic runs
random_seed_number=0
np.random.seed(random_seed_number)
torch.manual_seed(random_seed_number)
# make cudnn backend deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False