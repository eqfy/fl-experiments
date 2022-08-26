import itertools
import json
import pickle
import random
import pandas as pd
import numpy as np

"""
Utility function to combine FedScale client computation data with Measurement Lab client bandwidth data
"""




df = pd.read_csv('data/dataset_speedtest_ookla_full.csv')
df4G_LTE = df[df['radio_type'] == '4G-LTE']
df5G_NSA_mmWave = df[df['radio_type'] == '5G-NSA-mmWave']
df5G_NSA_LowBand = df[df['radio_type'] == '5G-NSA-Low-Band']
df5G_SA_LowBand = df[df['radio_type'] == '5G-SA-Low-Band']

df5G_Only = df[df['radio_type'] != '4G-LTE']
df5G_Only = df5G_Only[df5G_Only.columns[1:3]]
df5G_Only["min"] = df5G_Only.min(axis=1)
df5G_Only = df5G_Only.sort_values(by="min")

# print(df5G_Only)
# print(df5G_Only)
np5G_Only = np.array(df5G_Only)
np5G_Only *= 1000

print(np5G_Only[-20:])
# print(np5G_Only)
# print(np5G_Only[0], np5G_Only[0][0])
ul_dl_5G = [{"dl_kbps": i[0], "ul_kbps": i[1]} for i in np5G_Only]

# print(ul_dl_5G[:20])



# Data from https://www.measurementlab.net/tests/ndt/
# ul_dl_mlab_ndt = []
# with open('data/ul_dl_NA_2022.06.A.json') as fin:
#     ul_dl_mlab_ndt = json.load(fin)
# ul_dl_mlab_ndt = [{'dl_kbps': float(i['dl_mbps']) * 1000, 'ul_kbps': float(i['ul_mbps']) * 1000} for i in ul_dl_mlab_ndt if float(i['ul_mbps']) < 0.4209 or float(i['ul_mbps']) > 0.4210]
# ul_dl_mlab_ndt = sorted(ul_dl_mlab_ndt, key=lambda i: i['ul_kbps'] + i['dl_kbps'])
# # ul_dl_mlab_ndt = sorted(ul_dl_mlab_ndt, key=lambda i: min(i['ul_kbps'], i['dl_kbps']))

# size = len(ul_dl_mlab_ndt)
# change_size = int(size * 0.025)
# ul_dl_mlab_ndt = ul_dl_mlab_ndt[change_size: size - change_size]

# print(ul_dl_mlab_ndt[0], ul_dl_mlab_ndt[-1])
# print(ul_dl_mlab_ndt[:20])
# abc = sorted(ul_dl_mlab_ndt.copy(), key=lambda i: i["ul_kbps"])
# print(abc[0], abc[-1])
# print(abc[:20])
# slower_than_n = 0
# for i in abc:
#     if i["ul_kbps"] > 75:
#         break
#     slower_than_n += 1

# print(f"{slower_than_n}  {slower_than_n / len(abc)}")

# Data from FedScale repo
fedscale_client = {}
with open('data/client_device_capacity', 'rb') as fin:
    fedscale_client = pickle.load(fin)
fedscale_client_len = len(fedscale_client)
fedscale_client = dict(itertools.islice(fedscale_client.items(), len(ul_dl_5G)))
fedscale_client = [x for _, x in fedscale_client.items()]
fedscale_client = sorted(fedscale_client, key=lambda i: float(i['communication']))

# Combine the two client capacity data
print(ul_dl_5G[-20:])
merged_capacity = [fs | ul_dl for fs, ul_dl in zip(fedscale_client, ul_dl_5G)]
print(merged_capacity[-20:])
random.shuffle(merged_capacity)

# FedScale's data is 500,000 entries, hence we are extending merged_capacity by 490,000 entries by randomly selecting from our existing 10,000
merged_capacity.extend(random.choices(merged_capacity, k=fedscale_client_len - len(merged_capacity)))
merged_capacity = {i: cl for i, cl in enumerate(merged_capacity)}

floc = 'data/client_device_capacity_ul_dl_5G'
with open(floc, 'wb') as fout:
    pickle.dump(merged_capacity, fout)
print(f"Successfully dumped {len(merged_capacity)} client device capacity with upload and download bandwidth to {floc}")

