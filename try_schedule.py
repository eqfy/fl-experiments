
# compute time to finish for X% of clients
compute_ttf_75 = 0.110  # s
compute_ttf_90 = 0.170  # s

# minimum network throughput for X% of clients
throughput_50 = 10000  # kbps
throughput_75 = 1000   # kbps
throughput_90 = 110    # kbps

# model size
model_size = 28 * 1024 * 8  # 28MB in kb

# downstream size (converted from MB to kb)
downstream_sizes_c20 = [i * 8 * 1024 for i in [0, 2.5, 2.9, 3.0, 3.5, 3.9,
                                               4.2, 4.5, 4.8, 4.9, 5.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6]]

# compression ratio
compress_rate = 0.2

# availability
availability = 0.95


def getPrefetchOverhead(downstream, upstream, compute, rounds):
    full_update_time = model_size / downstream
    if rounds == 0:
        return full_update_time

    # total_download_time = 0
    # for i in range(rounds):
    #     total_download_time += getSingleDownloadTime(downstream, i)

    total_download_time = getSingleDownloadTime(downstream, rounds)
    for i in range(1, rounds):
        total_download_time += getSingleDownloadTime(downstream, 1)

    total_round_time = 0
    for i in range(rounds):
        total_round_time += getNormalRoundTime(
            downstream, upstream, compute)

    # addition time is an estimate of how long it takes to add updates
    addition_time = compute / 10

    return full_update_time + total_download_time + addition_time * rounds - \
        total_round_time


def getNormalRoundTime(downstream, upstream, compute):
    return downstream_sizes_c20[1] / downstream + \
        compute + (compress_rate * model_size) / upstream


def getSingleDownloadTime(downstream, round):
    if round >= len(downstream_sizes_c20):
        return downstream_sizes_c20[-1] / downstream
    return downstream_sizes_c20[round] / downstream


def getMinPrefetchRounds(downstream, upstream, compute):
    for i in range(20):
        if getPrefetchOverhead(downstream, upstream, compute, i) <= 0:
            print(
                f"Prefetch {i} epochs to achieve single round update latency\nWith downstream: {downstream}kbps upstream: {upstream}kbps and compute time: {compute}s")
            return
    print("Number of prefetch rounds exceeds 20")


getMinPrefetchRounds(throughput_50, throughput_50,
                     compute_ttf_75)  # down == up
getMinPrefetchRounds(throughput_75, throughput_90, compute_ttf_75)  # down > up
getMinPrefetchRounds(throughput_90, throughput_75, compute_ttf_75)  # down < up

# for i in range(100):
#     print(i, " ", getPrefetchOverhead(
#         throughput_90, throughput_75, compute_ttf_75, i))

"""
Remarks:
Assuming uniform network throughput and compute time for each client:
Compression ratio: 20%

If downstream and upstream network throughput is the same, then 5 rounds is sufficient. The number of rounds decreases as downstream throughput increases compared to upstream throughput
Compute time for each client does not have a large effect on the number of prefetch rounds

Question:
Availability?
    - What is a reasonable assumption? (95%)
Stragglers?

Non-uniform distribution of network throuput and compute time?

Maybe we do not need entire downstream for presending
Presend X rounds

tradeoff - presend more rounds -> less bandwidth usage per round and maybe slightly larger model size
tradeoff - availability -> more over sampling

how much does runtime change and downstream change as we skip more rounds
"""
