
# This file is borrowed from https://github.com/Xu-Jingyi/FedCorr/blob/main/util/sampling.py

import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)] # initial user and index for whole dataset
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs)-dict_users[i])

    for key in dict_users.keys():
        dict_users[key] = list(dict_users[key])
    return dict_users


