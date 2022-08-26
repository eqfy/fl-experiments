import itertools
import json
import pickle
import random
import pandas as pd
import numpy as np

"""
Utility function to combine FedScale client computation data with Measurement Lab client bandwidth data
"""


# Each step being 50MBPs
bucket_GCP_private = np.array([2,3,3,3,24,25,35,8,0,0])
bucket_GCP_public = np.array([2,7,3,3,27,37,36,22,14,4])
bucket_GCP_combined = bucket_GCP_private + bucket_GCP_public
normalized_bucket = bucket_GCP_combined / np.sum(bucket_GCP_combined)
avg_upload = 90

# cum = 0
# cum_bucket = []
# for i in normalized_bucket:
#     cum += i
#     cum_bucket
print(normalized_bucket)


ul_dl = []
for i in range(100):
    curr_rand = random.random() 
    bracket = 0
    for prob in normalized_bucket:
        # print(curr_rand, prob)
        if curr_rand <= prob:
            break
        curr_rand -= prob
        bracket += 50
    # print("\n\n")
    dl = (bracket + 50 * random.random()) * 1000
    ul = (np.random.normal(avg_upload, 7)) * 1000
    ul_dl.append({"dl_kbps": dl, "ul_kbps": ul})


ul_dl.sort(key=lambda i: i["dl_kbps"])
    
# print(ul_dl_5G[:20])

# # Data from FedScale repo
fedscale_client = {}
with open('data/client_device_capacity', 'rb') as fin:
    fedscale_client = pickle.load(fin)
fedscale_client_len = len(fedscale_client)
fedscale_client = dict(itertools.islice(fedscale_client.items(), len(ul_dl)))
fedscale_client = [x for _, x in fedscale_client.items()]
fedscale_client = sorted(fedscale_client, key=lambda i: float(i['communication']))

# Combine the two client capacity data
print(ul_dl[-20:])
merged_capacity = [fs | ul_dl for fs, ul_dl in zip(fedscale_client, ul_dl)]
print(merged_capacity[-20:])
random.shuffle(merged_capacity)

# FedScale's data is 500,000 entries, hence we are extending merged_capacity by 490,000 entries by randomly selecting from our existing 10,000
merged_capacity.extend(random.choices(merged_capacity, k=fedscale_client_len - len(merged_capacity)))
merged_capacity = {i: cl for i, cl in enumerate(merged_capacity)}

floc = 'data/client_device_capacity_ul_dl_GCP'
with open(floc, 'wb') as fout:
    pickle.dump(merged_capacity, fout)
print(f"Successfully dumped {len(merged_capacity)} client device capacity with upload and download bandwidth to {floc}")

