import os
import pickle
from matplotlib import pyplot as plt

import numpy as np


def load_client_profile(file_path):
    # load client profiles
    global_client_profile = {}
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fin:
            # {clientId: [computer, bandwidth]}
            global_client_profile = pickle.load(fin)

    return global_client_profile


client_profiles = load_client_profile("client_device_capacity")
client_profiles_list = [list(v.values())
                        for v in client_profiles.values()][:1000]
client_profiles_arr = np.array(client_profiles_list)
plt.scatter(x=client_profiles_arr[:, 0], y=client_profiles_arr[:, 1])
plt.show()
