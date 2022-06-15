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


client_profiles = load_client_profile("data/client_device_capacity")
client_profiles_list = [list(v.values())
                        for v in client_profiles.values()][:10000]
client_profiles_arr = np.array(client_profiles_list)
client_profiles_arr[:, 1] = client_profiles_arr[:, 1] / 10000
plt.scatter(x=client_profiles_arr[:, 0], y=client_profiles_arr[:, 1])
plt.xlabel("Compute time (ms)")
plt.ylabel("Bandwidth (Mbps)")
plt.title("FedScale client capacity")
# plt.show()
plt.savefig("figs/fedscale_client_capacity.png")
