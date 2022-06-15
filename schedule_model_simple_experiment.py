import enum
import json
from math import ceil
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


"""
TODO: Vary skipped rounds find 1. runtime 2. minimum downstream bandwidth needed
"""


def find_min_downfactor(compute, downstream, skipped_round, label, model_sizes=model_sizes_c20, compress_rate=.2, upfactor=.3, upstream=-1, addfactor=.2, speedunit="MB/s"):

    if skipped_round > 19 and (skipped_round not in [50, 100, 200]):
        print("skipped round invalid")
        return

    if upstream > 0:
        upfactor = upstream / downstream

    if speedunit == "Mbps":
        downstream /= 8
        upstream /= 8

    single_upload_size = compress_rate * full_model_size

    prefactors = []
    total_downstreams = []
    prefetch_rounds = range(1, 18)
    for prefetch_round in prefetch_rounds:
        # skipped_round == 0 means no prefetching at all
        if skipped_round - prefetch_round < 0:
            break

        first_fetch_size = model_sizes[skipped_round - prefetch_round]
        total_downstream = first_fetch_size + \
            model_sizes[prefetch_round]

        print(first_fetch_size, "\t", model_sizes[prefetch_round])

        addition_time = compute * addfactor

        # prefactor is the min percentage of downstream that we need to use to achieve speedup
        prefactor = (first_fetch_size + model_sizes[prefetch_round]) / \
            (prefetch_round * (model_sizes[1] +
                               single_upload_size / upfactor + downstream * compute) - downstream * addition_time)

        regular_time = prefetch_round * \
            (model_sizes[1] / downstream + compute +
             single_upload_size / (downstream * upfactor))

        prefetch_time = (
            first_fetch_size + model_sizes[prefetch_round]) / (downstream * prefactor) + addition_time

        print(f"Prefetch {prefetch_round} - skip {skipped_round} rounds, compute time {compute}s, downstream {downstream}MB/s:\n\
        \ttotal downstream: {total_downstream}MB\n\
        \tminimal downstream factor: {prefactor}\n\
        \twith reg time {regular_time}s and prefetch time {prefetch_time}s")

        prefactors.append(prefactor)
        total_downstreams.append(total_downstream)

    line_type = "--" if label == "stc" else ""
    # plt.plot(prefetch_rounds, prefactors, line_type,
    #          label=f"{label} compute{compute}s down{downstream:.2f}MB/s up{downstream * upfactor:.2f}MB/s skip{skipped_round}")
    plt.plot(prefetch_rounds, total_downstreams, line_type,
             label=f"{label} compute{compute}s down{downstream}MB/s skip{skipped_round}")


def find_min_prefetch(downstream_c, downstream, upstream, skipped_round=50, compute=10., label='None', model_sizes=model_sizes_c20, full_model_size=full_model_size, compress_rate=.2, upfactor=.3, speedunit='MB/s'):
    if skipped_round == 1:
        return 0
    elif skipped_round < 0 or (skipped_round > 19 and (skipped_round not in [50, 100, 200])):
        print("skipped round invalid")
        return -1

    if upstream > 0:
        upfactor = upstream / downstream

    if speedunit == "Mbps":
        downstream /= 8
        upstream /= 8

    for R in range(1, 100):
        minimum_R = (model_sizes[skipped_round - 1 - R] + model_sizes[R - 1]) / downstream_c / (
            model_sizes[1] / downstream + compute + compress_rate * full_model_size / downstream / upfactor)
        if minimum_R > 0:
            return ceil(minimum_R), minimum_R

    print('cannot find valid number of prefetch rounds')
    return -1


def find_min_prefetch_treg(downstream_c, t_reg, current_round, last_sync_round=-1, skipped_round=-1, model_sizes=model_sizes_c20, full_model_size=full_model_size):
    if last_sync_round >= 0:
        skipped_round = current_round - last_sync_round
    # if skipped_round < 0 or (skipped_round > 19 and (skipped_round not in [50, 100, 200])):
    #     print(f"skipped round {skipped_round} is invalid")
    #     sys.exit(1)

    # print(
    #     f"find_min_prefetch_treg down{downstream_c} curr round{current_round} lr{last_sync_round} sr{skipped_round}")
    for prefetch_round in range(1, 100):
        if current_round - prefetch_round < 0:
            return 100, -1

        # print(
        #     f"\tcurr round {current_round} skipped {skipped_round} prefetch {prefetch_round} lmodelsize{model_sizes[skipped_round - 1 - prefetch_round]} smodelsize{model_sizes[prefetch_round-1]} used t_reg{np.min(t_reg[current_round - prefetch_round:current_round])}")

        # print(current_round - prefetch_round, current_round)
        minimum_R = (model_sizes[skipped_round - 1 - prefetch_round]) / \
            downstream_c / \
            np.min(t_reg[current_round - prefetch_round:current_round])

        # print(f"\tminr {minimum_R} prer {prefetch_round}")
        if prefetch_round >= minimum_R:
            return prefetch_round, minimum_R

    print('cannot find valid number of prefetch rounds, defaulting to 100')
    print(
        f"downstream {downstream_c} t_reg{np.min(t_reg[current_round - prefetch_round:current_round])} curr_r{current_round} skip_r{skipped_round} model_size{model_sizes[skipped_round - 1 - prefetch_round]}")
    return 100, -1


def find_single_T_reg(downstream, upstream, model_sizes=model_sizes_c20, compute=10, compress_rate=0.2):
    return model_sizes[1] / downstream + compute + compress_rate * full_model_size / upstream


def getClientDist():
    f = open('../2021.05.01.datapoints.json')
    data = json.load(f)

    client_dist = []

    for i, d in enumerate(data):
        client_dist.append(
            [i, float(d['dl_mbps']) / 8, float(d['ul_mbps']) / 8, max(np.random.normal(10, 3), 0.5)])

    client_dist = np.array(client_dist)
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


def simulation(target_round):
    r = 0
    total_time = 0
    total_download = 0
    prefetch_dict = np.zeros(101, dtype=int)

    client_dist = getClientDist()
    num_client = client_dist.shape[0]
    client_last_round = np.zeros(num_client)
    T_regs = np.zeros(target_round)

    # sticky only sample for 10 rounds
    sampled = client_dist[np.random.choice(num_client, 28, False)]
    t_reg = find_single_T_reg(
        sampled[0][1], sampled[0][2], compute=sampled[0][3])
    for i in sampled:
        temp = find_single_T_reg(i[1], i[2], compute=sampled[0][3])
        if temp < t_reg:  # FIXME maybe actually use max here
            t_reg = temp

    T_regs[:10] = t_reg
    r += 10
    client_last_round[sampled[:0].astype(int)] = 10

    while r < target_round:
        # normal sticky sampling (with replacement)
        # Not entirely correct here but for demonstration purposes
        sampled = sampled[np.random.choice(28, 21, False)]

        new_sampled = client_dist[np.random.choice(num_client, 7, False)]
        sampled = np.append(
            sampled, new_sampled, axis=0)

        largest_round = 0
        for i in new_sampled:
            t, tfactor = find_min_prefetch_treg(
                i[1], T_regs, r, last_sync_round=int(client_last_round[int(i[0])]))
            largest_round = max(largest_round, t)

            if largest_round == 20:
                print(i)

        prefetch_dict[largest_round] += 1

        # Find new lowest t_reg
        t_reg = find_single_T_reg(
            sampled[0][1], sampled[0][2], compute=sampled[0][3])
        for i in sampled:
            temp = find_single_T_reg(i[1], i[2], compute=sampled[0][3])
            t_reg = min(temp, t_reg)  # FIXME maybe actually use max here

        T_regs[r] = t_reg
        r += 1

    # print(prefetch_dict)
    # print(T_regs)
    # print(model_sizes_c20)
    pdf = prefetch_dict / np.sum(prefetch_dict)
    cdf = np.cumsum(pdf)
    plt.plot(pdf)
    plt.plot(cdf)
    plt.ylim(0, 1.0)
    # plt.xlim(1, 30)
    # plt.xticks(np.arange(1, 30, step=1))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.grid()
    plt.show()


simulation(400)


"""
Simple experiments that fix one parameter and vary other parameters
"""


def simple_experiments():
    client_dist = getClientDist()

    # Vary compute time (s)
    compute_times = [1.25, 2.5, 5, 10, 20, 40, 80]
    rounds_needed = []
    round_factor = []
    for time in compute_times:
        r, factor = find_min_prefetch(downstream_c=4.21,
                                      downstream=161.83, upstream=27.84, compute=time, speedunit="Mbps")
        rounds_needed.append(r)
        round_factor.append(factor)

    plt.figure()
    plt.plot(compute_times, rounds_needed)
    plt.plot(compute_times, round_factor)
    plt.xlabel('Downstream (Mbps)')
    plt.ylabel('Required prefetch round')
    plt.title("Compute time vs required prefetch round")
    plt.savefig('compute_time_prefetch_round')

    # Ultimate goal is a CDF showing the % of training rounds that need X rounds of prefetching


# simple_experiments()


# Tradeoffs
# A big factor
# prefetching only for slow clients?
# finish simulation
# work on prefetching implementation
# AB Prefetch
# Prefetch?
