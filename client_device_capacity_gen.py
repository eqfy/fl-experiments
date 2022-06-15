import itertools
import json
import pickle

"""
Utility function to combine FedScale client computation data with Measurement Lab client bandwidth data
"""

"""Data from FedScale repo"""
fedscale_client = {}
with open('data/client_device_capacity', 'rb') as fin:
    fedscale_client = pickle.load(fin)
fedscale_client = dict(itertools.islice(fedscale_client.items(), 10000))
fedscale_client = [x for _, x in fedscale_client.items()]
fedscale_client = sorted(fedscale_client, key=lambda i: float(i['communication']))

"""Data from https://www.measurementlab.net/tests/ndt/"""
ul_dl_mlab_ndt = []
with open('data/ul_dl_capacity.json') as fin:
    ul_dl_mlab_ndt = json.load(fin)
ul_dl_mlab_ndt = [{'dl_kbps': float(i['dl_mbps']) * 1000, 'ul_kbps': float(i['ul_mbps']) * 1000} for i in ul_dl_mlab_ndt]
ul_dl_mlab_ndt = sorted(ul_dl_mlab_ndt, key=lambda i: i['ul_kbps'] + i['dl_kbps'])

merged_capacity = {i: fs | ul_dl for i, (fs, ul_dl) in enumerate(zip(fedscale_client, ul_dl_mlab_ndt))}

floc = 'data/client_device_capacity_ul_dl'
with open(floc, 'wb') as fout:
    pickle.dump(merged_capacity, fout)
print(f"Successfully dumped {len(merged_capacity)} client device capacity with upload and download bandwidth to {floc}")
