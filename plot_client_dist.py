import json
import numpy as np
from matplotlib import pyplot as plt

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def getClientDist():
    f = open('data/ul_dl_NA_2022.06.A.json')
    data = json.load(f)

    client_dist = []

    # index, download, upload, compute
    for i, d in enumerate(data):
        client_dist.append(
            [i, float(d['dl_mbps']), float(d['ul_mbps'])])

    client_dist = np.array(client_dist)

    # Remove strange test cases where upload is consistently between (0.4209, 0.4210)
    mask = ~((client_dist[:,2] - 0.4209 > 0) & (client_dist[:,2] - 0.4210 < 0))
    client_dist = client_dist[mask, :]
    client_dist = client_dist[:1000]

    ul_over_dl = client_dist[:,2] / client_dist[:,1]
    ul_over_dl_cleaned = reject_outliers(ul_over_dl)
    print(np.sum(ul_over_dl), len(ul_over_dl), len(ul_over_dl[ul_over_dl < 1.0]))
    np.mean(ul_over_dl)
    print(np.mean(ul_over_dl), np.median(ul_over_dl), np.quantile(ul_over_dl, 0.8))

    # Remove slowest or keep fastest
    # client_dist = client_dist[np.argsort(client_dist[:, 1])]
    # client_dist = client_dist[8000:]

    # print(client_dist)
    # plt.figure()
    # plt.scatter(np.arange(len(ul_over_dl_cleaned)), ul_over_dl_cleaned, s=2)
    # plt.ylim(0, 2)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.axes().set_aspect('equal', 'datalim')

    x = client_dist[:, 1]
    y = client_dist[:, 2]

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=14)
    plt.plot(np.arange(1000), "r--", lw=4)
    plt.xlabel('Downstream (Mbps)', fontsize=20)
    plt.ylabel('Upstream (Mbps)', fontsize=20)
    plt.xticks(fontsize=15., y=0.005)
    plt.yticks(fontsize=15., x=0.005)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10**-0.5, 10**3)
    plt.ylim(10**-0.5, 10**3)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Client Bandwidth (NA 2022-06-01 -- 2022-07-01)')
    plt.grid()
    plt.savefig('figs/client_bandwidth_dist.eps')
    plt.savefig('figs/client_bandwidth_dist.png')
    # plt.show()
    return client_dist


getClientDist()