import json
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit


def showQuartileDist():
    client_dist_CA = [0.009, 1.317, 4.211, 12.789, 23.766, 38.826,
                      57.816, 83.613, 123.352, 215.936, 346.27, 457.586, 669.12]
    client_dist_x = [0.01, 0.05, 0.1, 0.2, 0.3,
                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    plt.plot(client_dist_x, client_dist_CA)
    plt.grid()
    plt.xlabel("Quartile")
    plt.ylabel("Bandwidth (Mbps)")
    plt.show()


def lognorm_fit(x, mu, sigma):

    return (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))


def showClientDist():
    f = open('../2022.05.01.CA_histogram_3.json')

    data = json.load(f)

    ul_counts_CA, dl_counts_CA, dist_x = [], [], []

    dist_x.append(0.1)
    for bucket in data:
        ul_counts_CA.append(int(bucket['ul_samples_bucket']))
        dl_counts_CA.append(int(bucket['dl_samples_bucket']))
        dist_x.append(float(bucket['bucket_max']))

    # print(dist_x, ul_dist_CA)

    # plt.hist(ul_dist_CA, dist_x)
    # plt.hist(dist_x[:-1], dist_x, weights=ul_counts_CA,
    #          cumulative=True, density=True)
    # plt.hist(dist_x[:-1], dist_x, weights=dl_counts_CA,
    #          cumulative=True, density=True)

    counts, bins, _ = plt.hist(
        dist_x[:-1], dist_x, weights=ul_counts_CA, alpha=0.5, label='upload',
        # density=True,
        cumulative=True,
        stacked=True
    )
    # print(counts, bins)
    counts, bins, _ = plt.hist(
        dist_x[:-1], dist_x, weights=dl_counts_CA, alpha=0.5, label='download',
        # density=True,
        cumulative=True
        # stacked=True
    )

    """
    # restore data from histogram: counts multiplied bin centers
    restored = [[d]*int(counts[n])
                for n, d in enumerate((bins[1:]+bins[:-1])/2)]
    # flatten the result
    restored = [item for sublist in restored for item in sublist]

    print(stats.lognorm.fit(restored, floc=0))
    dist = stats.lognorm(*stats.lognorm.fit(restored, floc=0))
    # y = dist.pdf(dist_x)
    y = dist.pdf(dist_x)
    y = y/y.max()
    y = y * counts.max()

    plt.plot(dist_x, y, 'r')
    """

    # popt, pcov = curve_fit(lognorm_fit, xdata=dist_x[1:], ydata=dl_counts_CA)
    # plt.plot(dist_x, lognorm_fit(dist_x, *popt), 'r-',
    #          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

    plt.grid()
    plt.xscale('log')
    plt.xlabel("Bucket bandwidth (Mbps)")
    plt.ylabel("Client count")
    plt.legend()
    plt.title('Client bandwidth distribution (Canada 2022.05.01 Total 28921)')
    plt.savefig("figs/client_bandwidth_cdf.png")
    plt.show()


showClientDist()
