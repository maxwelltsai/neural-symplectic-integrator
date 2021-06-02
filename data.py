

import os, sys
import h5py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''
    path = os.path.join(save_dir, 'train_test.h5')
    with h5py.File(path, 'r') as h5f:
        data = {}
        for dset in h5f.keys():
            data[dset] = h5f[dset][()]

    print("Successfully loaded data from {}".format(path))

    return data
