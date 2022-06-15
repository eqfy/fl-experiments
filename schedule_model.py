import enum
from hashlib import new
import json
from math import ceil
from pydoc import cli
from random import sample
import sys
from time import time
from matplotlib import pyplot as plt
import numpy as np

# compute time to finish for X % of clients
compute_ttf_75 = 0.110  # s
compute_ttf_90 = 0.170  # s

# minimum network throughput for X% of clients
throughput_50 = 10000  # kbps
throughput_75 = 1000   # kbps
throughput_90 = 110    # kbps

# model sizes (in MB)
full_model_size = 28.

# partial model sizes for download
model_sizes_c10 = [0] * 400
model_sizes_c20 = [0] * 400
model_sizes_stc = [0] * 400

# model sizes that need to be downloaded after skipping X rounds (in MB)
model_sizes_c10[:20] = [0, 2.5, 2.9, 3.0, 3.5, 3.9,
                        4.2, 4.5, 4.8, 4.9, 5.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6]
model_sizes_c20[:20] = [0, 6., 6.4, 7., 7.4, 7.9, 8.3, 8.7, 9.1,
                        9.5, 10., 12.7, 13., 13.3, 13.6, 13.9, 14.2, 14.5, 14.7, 14.9]
model_sizes_stc[:20] = [0, 6., 10., 13., 16., 17.7, 19., 20.7,
                        21.6, 22.6, 23, 23.5, 24., 24.5, 24.7, 25., 25.2, 25.6, 26.]

# Add some more datapoints
# model_sizes_c20[50] = 22.
# model_sizes_c20[100] = 25.
# model_sizes_c20[200] = 26.5

model_sizes_c20[20:40] = [15.0, 15.2, 17.2, 17.4, 17.5, 17.6, 17.7, 17.8,
                          17.9, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 21.2, 21.3, 21.4]
model_sizes_c20[40:60] = [22.1, 22.16, 22.22, 22.28, 22.34, 22.4, 22.43,
                          22.45, 22.47, 22.94, 22.99, 23.03, 23.06, 23.08, 23.11, 23.13, 23.15, 23.16, 23.17, 23.18]
model_sizes_c20[60:80] = [23.5, 23.51, 23.52, 23.53, 23.54, 23.55, 23.6,
                          23.61, 23.62, 23.63, 23.65, 23.69, 23.72, 23.81, 23.85, 23.89, 23.92, 23.95, 24., 24.05]
model_sizes_c20[80:100] = [24.2, 24.25, 24.3, 24.36, 24.4, 24.51, 24.56,
                           24.58, 24.6, 24.7, 24.72, 24.74, 24.76, 24.78, 24.8, 24.9, 24.92, 24.94, 24.96, 24.98]
model_sizes_c20[100:150] = [25.5] * 50
model_sizes_c20[150:200] = [26] * 50
model_sizes_c20[200:] = [27] * 200


model_sizes_stc[50] = 26
model_sizes_stc[100] = 27.9
model_sizes_stc[200] = 28
model_sizes_stc[80:100] = [27.6, 27.6, 27.6, 27.6, 27.7, 27.7, 27.7,
                           27.7, 27.7, 27.7, 27.7, 27.7, 27.8, 27.8, 27.8, 27.8, 27.8, 27.8, 27.8, 27.8, 27.9]

# availability
availability = 0.95


def find_min_prefetch_treg(downstream_c, t_reg, current_round, last_sync_round=-1, skipped_round=-1, model_sizes=model_sizes_c20, full_model_size=full_model_size):
    if last_sync_round >= 0:
        skipped_round = current_round - last_sync_round

    for prefetch_round in range(1, 100):
        if current_round - prefetch_round < 0:
            print('cannot find valid number of prefetch rounds, defaulting to 100')
            return 100, 27, -1

        minimum_R = (model_sizes[skipped_round - 1 - prefetch_round]) / \
            downstream_c / \
            np.min(t_reg[current_round - prefetch_round:current_round])

        if prefetch_round >= minimum_R:
            #     print(
            #         f"prefetch {prefetch_round} missed {skipped_round - 1 - prefetch_round} remaining missed {prefetch_round}")
            return prefetch_round, model_sizes[skipped_round - 1 - prefetch_round] + model_sizes[prefetch_round], minimum_R

    print(
        f"downstream {downstream_c} t_reg{np.min(t_reg[current_round - prefetch_round:current_round])} curr_r{current_round} skip_r{skipped_round} model_size{model_sizes[skipped_round - 1 - prefetch_round]}")
    return 100, 27, -1


def find_single_T_reg(downstream, upstream, model_sizes=model_sizes_c20, compute=10, compress_rate=0.2):
    return model_sizes[1] / downstream + compute + compress_rate * full_model_size / upstream


def find_no_prefetch_time(downstream, upstream, compute, missed_round, model_sizes=model_sizes_c20, compress_rate=0.2):
    if missed_round < 1:
        missed_round = 1
    return model_sizes[missed_round] / downstream + compute + compress_rate * full_model_size / upstream


def getClientDist(remove_rate=0):
    f = open('../2021.05.01.datapoints.json')
    data = json.load(f)

    client_dist = []

    # index, download, upload, compute
    for i, d in enumerate(data):
        client_dist.append(
            [i, float(d['dl_mbps']) / 8, float(d['ul_mbps']) / 8, np.random.lognormal(0, 1)])

    client_dist = np.array(client_dist)
    client_dist = client_dist[np.argsort(client_dist[:, 1])]

    # Remove slowest % of client
    full_size = len(client_dist)
    removed_size = int(full_size * remove_rate)
    client_dist = client_dist[:full_size - removed_size]
    client_dist[:, 0] = np.arange(full_size - removed_size)

    # print(client_dist.shape, client_dist[:, 0].shape, client_dist[:, 1].shape)

    x = client_dist[:, 0]
    y = client_dist[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    # print(A)
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # print(m, c)
    sorted_x = np.sort(x)

    # plt.figure()
    # plt.plot(sorted_x, m*sorted_x + c, 'r')
    # plt.scatter(x, y)
    # plt.xlabel('Downstream (Mbps)')
    # plt.ylabel('Upstream (Mbps)')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Distribution of client downstream and upstream (Canada 2022-05-01)')
    # plt.savefig('figs/client_bandwidth_dist.png')
    # plt.show()
    return client_dist


def simulation(target_round, client_dist=getClientDist(), sample_size=28, change_size=7, compress_ratio=0.2):
    r = 0
    total_time = 0
    total_download = 0.
    total_time_nopre = 0.
    total_download_nopre = 0
    total_upload = 0.
    prefetch_dict = np.zeros(101, dtype=int)

    num_client = client_dist.shape[0]
    client_last_round = np.zeros(num_client, dtype=int)
    T_regs = np.zeros(target_round)

    # sticky only sample for 10 rounds
    sampled = client_dist[np.random.choice(num_client, sample_size, False)]
    t_reg = find_single_T_reg(
        sampled[0][1], sampled[0][2], compute=sampled[0][3])
    for i in sampled:
        temp = find_single_T_reg(i[1], i[2], compute=i[3])
        # t_reg = min(t_reg, temp)
        t_reg = max(t_reg, temp)

    T_regs[:10] = t_reg

    client_last_round[sampled[:, 0].astype(int)] = 10

    total_upload += full_model_size * 0.2 * sample_size * 10
    total_download += model_sizes_c20[1] * sample_size * 10
    total_download_nopre += model_sizes_c20[1] * sample_size * 10
    total_time_nopre += t_reg * 10
    r += 10

    while r < target_round:
        # normal sticky sampling (with replacement)
        # Not entirely correct here but for demonstration purposes
        sampled = sampled[np.random.choice(
            sample_size, sample_size - change_size, False)]

        new_sampled = client_dist[np.random.choice(
            num_client, change_size, False)]
        sampled = np.append(
            sampled, new_sampled, axis=0)

        largest_round = 0
        largest_download = 0
        for i in new_sampled:
            t,  down, tfactor = find_min_prefetch_treg(
                i[1], T_regs, r, last_sync_round=client_last_round[int(i[0])])
            largest_round = max(largest_round, t)
            largest_download = max(largest_download, down)

            total_download_nopre += model_sizes_c20[r -
                                                    client_last_round[int(i[0])]]
            total_download += down  # for non-uniform down, requires adjust prefetch for each client

        prefetch_dict[largest_round] += 1

        # Find new largest t_reg
        # t_reg = np.inf
        t_reg = 0
        for i in sampled:
            temp = find_single_T_reg(i[1], i[2], compute=i[3])
            # t_reg = min(temp, t_reg)  # FIXME maybe actually use max here
            t_reg = max(t_reg, temp)

        T_regs[r] = t_reg

        # Find compute time without prefetch
        t_nopre = 0
        for i in sampled:
            t_nopre = max(t_nopre, find_no_prefetch_time(
                i[1], i[2], i[3], r - client_last_round[int(i[0])], compress_rate=0.2))
        total_time_nopre += t_nopre

        # Set all sampled client's last round to r
        client_last_round[sampled[:, 0].astype(int)] = r

        # total_download += largest_download * change_size  # uniform download
        total_download += model_sizes_c20[1] * (sample_size - change_size)
        total_upload += full_model_size * 0.2 * sample_size

        total_download_nopre += model_sizes_c20[1] * \
            (sample_size - change_size)
        r += 1

    # print(prefetch_dict)
    # print(T_regs)
    # print(model_sizes_c20)

    total_time = np.sum(T_regs)
    time_ratio = total_time / total_time_nopre
    dl_ratio = total_download / total_download_nopre

    print(f"time: {total_time} time noprefetch: {total_time_nopre} time ratio: {time_ratio}\ndownload: {total_download} download noprefetch: {total_download_nopre}  download ratio: {dl_ratio}\nupload: {total_upload}")
    # print(client_dist[:, 1], total_time)
    # plt.plot(client_dist[:, 1])
    # plt.plot(client_dist[:, 2])

    pdf = prefetch_dict / np.sum(prefetch_dict)
    cdf = np.cumsum(pdf)
    # plt.plot(pdf, label="PDF")
    # plt.plot(cdf, label="CDF")
    # plt.ylim(0, 1.0)
    # plt.xlim(0, 25)
    # # plt.xticks(np.arange(1, 30, step=1))
    # plt.yticks(np.arange(0, 1.1, step=0.1))
    # plt.grid()
    # plt.xlabel("Minimum required prefetch rounds")
    # plt.ylabel("Percent of rounds")
    # plt.title(
    #     f"Scheduling simulation - nonuniform download")
    # plt.text(
    #     2, 0.8, f"{target_round} rounds, {len(client_dist)} clients, {sample_size} sample size, {change_size} change size")
    # plt.text(
    #     2, 0.75, f"Total time {total_time:.2f}s, total time no prefetch {total_time_nopre:.2f}")
    # plt.text(
    #     2, 0.7, f"Total download {total_download/1024:.2f}GB, total download no prefetch {total_download_nopre/1024:.2f}GB")
    # plt.text(
    #     2, 0.65, f"Time ratio {time_ratio:.2f} Download ratio {dl_ratio:.2f}")
    # plt.text(
    #     2, 0.6, f"Clients: Canada bandwidth + mock lognormal dist. compute time")
    # plt.legend()
    # plt.savefig('figs/prefetch_simu_nonuni_down.png')
    # plt.show()

    return total_time, total_download, time_ratio,  dl_ratio,  prefetch_dict


def simulation_oversample(target_round, client_dist=getClientDist(), sample_size=28, change_size=7, oversample=2):
    r = 0
    total_time = 0
    total_time_nopre = 0
    total_download = 0.
    total_download_nopre = 0
    total_upload = 0.
    prefetch_dict = np.zeros(101, dtype=int)

    num_client = client_dist.shape[0]
    client_last_round = np.zeros(num_client)
    T_regs = np.zeros(target_round)

    # sticky only sample for 10 rounds
    sampled = client_dist[np.random.choice(num_client, sample_size, False)]
    t_reg = find_single_T_reg(
        sampled[0][1], sampled[0][2], compute=sampled[0][3])
    # t_reg = 0
    for client in sampled:
        temp = find_single_T_reg(client[1], client[2], compute=client[3])
        # t_reg = min(t_reg, temp)
        t_reg = max(t_reg, temp)

    T_regs[:10] = t_reg

    client_last_round[sampled[:, 0].astype(int)] = 10

    total_upload += full_model_size * 0.2 * sample_size * 10
    total_download += model_sizes_c20[1] * sample_size * 10
    total_download_nopre += model_sizes_c20[1] * sample_size * 10
    total_time_nopre += t_reg * 10
    r += 10

    while r < target_round:
        # normal sticky sampling (with replacement)
        # Not entirely correct here but for demonstration purposes
        sampled = sampled[np.random.choice(
            sample_size, sample_size - change_size, False)]

        new_sampled = client_dist[np.random.choice(
            num_client, change_size + oversample, False)]

        round_numbers = []
        for i, client in enumerate(new_sampled):
            t,  down, tfactor = find_min_prefetch_treg(
                client[1], T_regs, r, last_sync_round=int(client_last_round[int(client[0])]))
            round_numbers.append([i, t])

            total_download_nopre += model_sizes_c20[r -
                                                    int(client_last_round[int(client[0])])]
            total_download += down  # for non-uniform down, requires adjust prefetch for each client

        round_numbers.sort()
        round_numbers = np.array(round_numbers)
        prefetch_dict[round_numbers[-oversample][1]] += 1
        new_sampled = new_sampled[round_numbers[:change_size, 0]]

        sampled = np.append(
            sampled, new_sampled, axis=0)
        # Find new largest t_reg
        # t_reg = np.inf

        t_reg = 0
        for client in sampled:
            temp = find_single_T_reg(client[1], client[2], compute=client[3])
            # t_reg = min(temp, t_reg)  # FIXME maybe actually use max here
            t_reg = max(t_reg, temp)

        T_regs[r] = t_reg

        # Find compute time without prefetch
        t_nopre = 0
        for i in sampled:
            t_nopre = max(t_nopre, find_no_prefetch_time(
                i[1], i[2], i[3], r - int(client_last_round[int(i[0])]), compress_rate=0.2))
        total_time_nopre += t_nopre

        # Set all sampled client's last round to r
        client_last_round[sampled[:, 0].astype(int)] = r

        # total_download += largest_download * change_size
        total_download += model_sizes_c20[1] * (sample_size - change_size)
        total_upload += full_model_size * 0.2 * sample_size
        total_download_nopre += model_sizes_c20[1] * \
            (sample_size - change_size)
        r += 1

    # print(prefetch_dict)
    # print(T_regs)
    # print(model_sizes_c20)

    total_time = np.sum(T_regs)
    time_ratio = total_time / total_time_nopre
    dl_ratio = total_download / total_download_nopre

    print(f"time: {total_time} time noprefetch: {total_time_nopre} time ratio: {time_ratio}\ndownload: {total_download} download noprefetch: {total_download_nopre}  download ratio: {dl_ratio}\nupload: {total_upload}")

    # print(client_dist[:, 1], total_time)
    # plt.plot(client_dist[:, 1])
    # plt.plot(client_dist[:, 2])

    pdf = prefetch_dict / np.sum(prefetch_dict)
    cdf = np.cumsum(pdf)
    # plt.plot(pdf, label="PDF")
    # plt.plot(cdf, label="CDF")
    # plt.ylim(0, 1.0)
    # plt.xlim(0, 10)
    # # plt.xticks(np.arange(1, 30, step=1))
    # plt.yticks(np.arange(0, 1.1, step=0.1))
    # plt.grid()
    # plt.xlabel("Minimum required prefetch rounds")
    # plt.ylabel("Percent of rounds")
    # plt.title(
    #     f"Scheduling simulation - oversample greedy {oversample}")
    # plt.text(
    #     2, 0.8, f"{target_round} rounds, {len(client_dist)} clients, {sample_size} sample size, {change_size} change size")
    # plt.text(
    #     2, 0.75, f"Total time {total_time:.2f}s, total time no prefetch {total_time_nopre:.2f}")
    # plt.text(
    #     2, 0.7, f"Total download {total_download/1024:.2f}GB, total download no prefetch {total_download_nopre/1024:.2f}GB")
    # plt.text(
    #     2, 0.65, f"Time ratio {time_ratio:.2f} Download ratio {dl_ratio:.2f}")
    # plt.text(
    #     2, 0.6, f"Clients: Canada bandwidth + mock lognormal dist. compute time")
    # plt.legend()
    # plt.savefig('figs/prefetch_simu_over_greedy.png')
    # plt.show()

    return total_time, total_download, time_ratio,  dl_ratio,  prefetch_dict


def run_simulation():
    clientDist = getClientDist()
    time_ratios, dl_ratios = [], []

    for i in range(100):
        total_time, total_download, time_ratio, download_ratio, prefectch_dict = simulation(
            400, client_dist=clientDist, sample_size=28, change_size=8)
        time_ratios.append(time_ratio)
        dl_ratios.append(download_ratio)

    print(
        f"Time ratio avg:{np.average(time_ratios):.2f} Download ratio avg:{np.average(download_ratio):.2f}")
    plt.figure()
    plt.scatter(time_ratios, dl_ratios)
    plt.xlabel("Time ratio: prefetch/non-prefetch")
    plt.ylabel("Download ratio: prefetch/non-prefetch")
    plt.title(
        f"Time and download ratio scatterplot\nTime ratio avg:{np.average(time_ratios):.2f} Download ratio avg:{np.average(download_ratio):.2f}")
    plt.savefig("figs/dl_time_ratio_scatter_greedy_oversample.png")
    plt.show()


run_simulation()

# simulation_oversample(400)
# Tradeoffs
# A big factor
# prefetching only for slow clients?
# finish simulation
# work on prefetching implementation
# AB Prefetch
# Prefetch?
