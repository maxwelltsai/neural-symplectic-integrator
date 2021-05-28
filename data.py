

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import to_pickle, from_pickle


##### LOAD OR SAVE THE DATASET #####
def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''

    path = '{}/{}-orbits-dataset.pkl'.format(save_dir, experiment_name)
    print('training data path: %s' % path)
    data = from_pickle(path)
    print("Successfully loaded data from {}".format(path))

    return data
