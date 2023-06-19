import os
import numpy as np
import random
import torch
from distutils.version import LooseVersion as Version


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        '''
        To get same results when sampling during different runs.
        If you are using cuDNN, you should set the deterministic behavior.
        This might make your code quite slow, but might be a good method to check your code and deactivate it later.
        '''
        torch.backends.cudnn.deterministic = True

    # if torch.__version__ <= Version("1.7"):
    #     torch.set_deterministic(True)
    # else:
    #     torch.use_deterministic_algorithms(True)