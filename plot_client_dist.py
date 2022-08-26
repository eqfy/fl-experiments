from cProfile import label
import json
import numpy as np
from matplotlib import pyplot as plt

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def plot_dist(client_dist):
    x = client_dist[:, 1]
    y = client_dist[:, 2]

    label_size = 19
    font2 = {
        'weight': 'normal',
        'size': label_size,
    }

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=14)
    ax.plot(np.arange(1000), "r--", linewidth=3)

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90, wspace=None, hspace=None)
    ax.set_xlabel("Download (Mbps)", font2)
    ax.set_ylabel("Upload (Mbps)", font2)
    ax.tick_params(axis='x', labelsize= label_size)
    ax.tick_params(axis='y', labelsize= label_size)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10**-0.5, 10**3)
    ax.set_ylim(10**-0.5, 10**3)
    plt.grid(linestyle="-.")
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)
    plt.savefig('figs/bandwidth_dist.pdf')
    plt.savefig('figs/bandwidth_dist.png')

def plot_cdf(client_dist):
    dls = np.sort(client_dist[:, 1])
    uls = np.sort(client_dist[:, 2])
    dls_cum = np.arange(len(dls)) / len(dls)
    uls_cum = np.arange(len(uls)) / len(uls)

    label_size = 19
    font2 = {
        'weight': 'normal',
        'size': label_size,
    }

    fig, ax = plt.subplots()
    ax.plot(uls,uls_cum, label="Upload")
    ax.plot(dls,dls_cum, "--", label="Download")
    
    ax.fill_between(uls, uls_cum, alpha=0.3)
    ax.fill_between(dls, dls_cum, alpha=0.6)

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90, wspace=None, hspace=None)
    ax.set_xlabel("Bandwidth (Mbps)", font2)
    ax.set_ylabel("Cumulative probability", font2)
    ax.tick_params(axis='x', labelsize= label_size)
    ax.tick_params(axis='y', labelsize= label_size)
    ax.set_xscale('log')
    ax.set_xlim(10**-0.5, 10**3)
    ax.set_xlim()
    ax.set_ylim(0, 1)
    ax.legend(fontsize=label_size)
    plt.grid(linestyle="-.")
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)
    plt.savefig('figs/bandwidth_cdf.pdf')
    plt.savefig('figs/bandwidth_cdf.png')



def getClientDist():
    # A: Timestamp 1 < x < 150 
    # B: Timestamp x < 1
    # C: Timestamp x < 20
    # D: Timestamp 0.1 < x < 20
    f = open('data/ul_dl_NA_2022.06.A.json')
    data = json.load(f)

    client_dist = []

    # index, download, upload, compute
    for i, d in enumerate(data):
        client_dist.append(
            [i, float(d['dl_mbps']), float(d['ul_mbps'])])

    client_dist = np.array(client_dist)

    # Remove strange test cases where upload is consistently between (0.4209, 0.4210)
    mask = ~((client_dist[:,2] > 0.4209) & (client_dist[:,2] < 0.4210))
    client_dist = client_dist[mask, :]
    mask1 = (client_dist[:,1] > 0.2) & (client_dist[:,2] > 0.2)
    client_dist = client_dist[mask1, :]
    client_dist = client_dist[:1000]

    ul_over_dl = client_dist[:,2] / client_dist[:,1]
    ul_over_dl_cleaned = reject_outliers(ul_over_dl)
    print(np.sum(ul_over_dl), len(ul_over_dl), len(ul_over_dl[ul_over_dl < 1.0]))
    np.mean(ul_over_dl)
    print(np.mean(ul_over_dl), np.median(ul_over_dl), np.quantile(ul_over_dl, 0.8))

    # Remove slowest or keep fastest
    # client_dist = client_dist[np.argsort(client_dist[:, 1])]
    # client_dist = client_dist[8000:]

    plot_dist(client_dist)
    plot_cdf(client_dist)

    # plt.show()
    return client_dist


getClientDist()