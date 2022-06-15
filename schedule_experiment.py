# compute time to finish for X % of clients
from matplotlib import pyplot as plt


compute_ttf_75 = 0.110  # s
compute_ttf_90 = 0.170  # s

# minimum network throughput for X% of clients
throughput_50 = 10000  # kbps
throughput_75 = 1000   # kbps
throughput_90 = 110    # kbps

# model sizes (in MB)
full_model_size = 28.

# model sizes that need to be downloaded after skipping X rounds (in MB)
model_sizes_c10 = [0, 2.5, 2.9, 3.0, 3.5, 3.9,
                   4.2, 4.5, 4.8, 4.9, 5.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6]
model_sizes_c20 = [0, 6., 6.4, 7., 7.4, 7.9, 8.3, 8.7, 9.1,
                   9.5, 10., 12.7, 13., 13.3, 13.6, 13.9, 14.2, 14.5, 14.8]
model_sizes_stc = [0, 6., 10., 13., 16., 17.7, 19., 20.7,
                   21.6, 22.6, 23, 23.5, 24., 24.5, 24.7, 25., 25.2, 25.6, 26.]

for _ in range(200):
    model_sizes_c10.append(0)
    model_sizes_c20.append(0)
    model_sizes_stc.append(0)

# Add some more datapoints
model_sizes_c20[50] = 22.
model_sizes_c20[100] = 25.
model_sizes_c20[200] = 26.5
model_sizes_stc[50] = 26
model_sizes_stc[100] = 27.9
model_sizes_stc[200] = 28

model_sizes_c20[80:100] = [23, 23.1, 23.2, 23.3, 23.4, 23.5, 23.6,
                           23.7, 23.8, 23.9, 24., 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9, 24.9]


model_sizes_stc[80:100] = [27.6, 27.6, 27.6, 27.6, 27.7, 27.7, 27.7,
                           27.7, 27.7, 27.7, 27.7, 27.7, 27.8, 27.8, 27.8, 27.8, 27.8, 27.8, 27.8, 27.8, 27.9]

# availability
availability = 0.95


"""
TODO: Vary skipped rounds find 1. runtime 2. minimum downstream bandwidth needed
"""


# def getLastRoundRuntime(sround, model_sizes):
#     return model_sizes[sroun]

def find_min_down(compute, model_sizes=model_sizes_c20, compress_rate=.2, upfactor=.2, addfactor=.1):
    single_upload_size = compress_rate * full_model_size
    for skipped_round in range(1, 19):
        # skipped_round == 1 means no prefetching at all
        total_fetch_amount = model_sizes[skipped_round]
        if skipped_round > 0:
            total_fetch_amount += model_sizes[1] * (skipped_round - 1)

        addition_time = compute * addfactor
        upfactor = 0.2  # upstream = downstream * upfactor
        min_down = (upfactor * model_sizes[skipped_round] - upfactor * model_sizes[1] - skipped_round *
                    single_upload_size) / (upfactor * (skipped_round * compute - addition_time*(skipped_round - 1)))

        regular_time = skipped_round * \
            (model_sizes[1] / min_down + compute +
             single_upload_size / (min_down * upfactor))

        prefetch_time = model_sizes[skipped_round] / min_down + \
            (model_sizes[1] / min_down +
             addition_time) * (skipped_round - 1.)

        print(f"Prefetch {skipped_round} rounds:\n\
        \ttotal downstream: {total_fetch_amount}MB\n\
        \tminimal downstream bandwidth: {min_down}MB/s\n\
        \twith reg time {regular_time} and prefetch time {prefetch_time}")


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

# find_min_down(100)
# find_min_downfactor(10, 1)

# find_min_downfactor(10, 0.01, 100, "c20")
# find_min_downfactor(10, 0.1, 100, "c20")


# find_min_downfactor(10, 0.5, 15, "c20")
find_min_downfactor(10, 0.5, 100, "c20")
# find_min_downfactor(20, 1, 100, "c20", upfactor=2)
# find_min_downfactor(20, 1, 100, "c20", upfactor=1)
# find_min_downfactor(20, 1, 100, "c20", upfactor=0.2)
# find_min_downfactor(10, 0.5, 15, "stc", model_sizes=model_sizes_stc)
# find_min_downfactor(10, 0.5, 50, "stc", model_sizes=model_sizes_stc)
find_min_downfactor(10, 0.5, 100, "stc", model_sizes=model_sizes_stc)

# Canada - May 13
# full_model_size = 2.8
# model_sizes_c20 = [i * 0.1 for i in model_sizes_c20]
# find_min_downfactor(10, 26.92, 100, "c20-CAN",
#                     upstream=7.47, speedunit="Mbps")
# find_min_downfactor(10, 35.11, 100, "c20-USA",
#                     upstream=11.81, speedunit="Mbps")
# find_min_downfactor(10, 17.31, 100, "c20-IND",
#                     upstream=5.61, speedunit="Mbps")

# find_min_downfactor(10, 0.8, 100, "c20-CRIT",
#                     upfactor=0.3)
# find_min_downfactor(10, 0.5, 100, "c20-LOW",
#                     upfactor=0.3)
# find_min_downfactor(10, 26.92, 100, "stc - Can",
#                     upstream=7.47, speedunit="Mbps",  model_sizes=model_sizes_stc)
# find_min_downfactor(10, 35.11, 100, "stc - USA",
#                     upstream=11.81, speedunit="Mbps",  model_sizes=model_sizes_stc)

# full_model_size = 28
# model_sizes_c20 = [i * 1 for i in model_sizes_c20]
# find_min_downfactor(10, 26.92, 100, "c20 - Test",
#                     upstream=7.47, speedunit="Mbps")
# find_min_downfactor(10, 26.92, 17, "c20 - Test",
#                     upstream=7.47, speedunit="Mbps")


plt.xticks([1, 5, 9, 13, 17])
plt.grid()
plt.legend()
plt.xlabel("prefetch round")
# plt.ylabel("minimum downstream bandwidth (%)")
# plt.savefig("figs/prefetch_round_vs_downstream_usage.png")
plt.ylabel("total downstream cost (MB)")
plt.savefig("figs/prefetch_round_vs_total_downstream_cost.png")
# plt.show()
plt.savefig("test.png")
