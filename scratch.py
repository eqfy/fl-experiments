# logging.info(sortedWorkersByCompletion)
# sticky_worker_index = []
# new_worker_index = []
# total_worker = self.args.total_worker
# for i in sortedWorkersByCompletion:
#     if sampledClientsReal[i] in self.curr_worker_pool:
#         sticky_worker_index.append(i)
#     else:
#         new_worker_index.append(i)

# logging.info(sticky_worker_index)

# logging.info(new_worker_index)

# dummy_index = []
# dummy_index.extend(sticky_worker_index[total_worker - 7:])
# dummy_index.extend(new_worker_index[7:])

# logging.info(dummy_index)

# sticky_worker_index = sticky_worker_index[:total_worker - 7]
# new_worker_index = new_worker_index[:7]

# top_k_index = []
# top_k_index.extend(sticky_worker_index)
# top_k_index.extend(new_worker_index)

# logging.info(top_k_index)

# round_duration = min(completionTimes[sticky_worker_index[-1]], completionTimes[new_worker_index[-1]])

# clients_to_run = [sampledClientsReal[k] for k in top_k_index]
# dummy_clients = [sampledClientsReal[k] for k in dummy_index]

# client_completion_times = [completionTimes[i] for i in top_k_index]
# return clients_to_run, dummy_clients, completed_client_clock, round_duration, client_completion_times


a = [2397, 11240, 5875, 781, 814, 7998, 5916, 7045, 8486, 5156, 1030, 8455, 1279, 393, 5594, 10012, 335, 4373, 867, 6046, 3291, 5723, 2496, 9868, 6031, 2069, 1537, 4026, 6125, 2847, 9888, 1117, 7819, 8285, 7929, 551, 3436, 4405, 7374, 2206, 8129, 5026, 1045, 4571, 5027, 7759, 8809, 2686, 3210, 94,
     290, 9886, 1770, 5643, 2198, 1460, 3569, 5362, 7947, 2199, 3181, 8373, 2200, 9835, 852, 2697, 8012, 88, 4424, 10404, 6084, 9160, 2473, 3442, 4043, 4984, 877, 10203, 9649, 7674, 11258, 9156, 9322, 2759, 10993, 1495, 7348, 6506, 3032, 2816, 2490, 4954, 2060, 8556, 3591, 4159, 8927, 5751, 10471, 1648]

b = [2397,
     11240,
     2496,
     5156,
     867,
     8455,
     7045,
     3291,
     1030,
     5875,
     4373,
     393,
     7998,
     335,
     814,
     10012,
     781,
     8486,
     5723,
     5916,
     1279,
     5594,
     3436,
     2069,
     6046,
     6031,
     1537,
     6125,
     7929,
     9868,
     1117,
     551,
     8285,
     2847,
     2206,
     9888,
     7819,
     4026,
     5026,
     7759,
     94,
     8809,
     5643,
     1045,
     4405,
     9886,
     4571,
     7374,
     1460,
     8129,
     290,
     5027,
     1770,
     7947,
     5362,
     8012,
     2199,
     3569,
     2198,
     10404,
     2686,
     88,
     6084,
     3210,
     3442,
     4043,
     9835,
     877,
     2200,
     10993,
     4424,
     9322,
     9649,
     2697,
     2473,
     9160,
     3032,
     2759,
     10203,
     4159,
     7674,
     1495,
     852,
     4984,
     4954,
     2060,
     8373,
     8556,
     3591,
     11258,
     2490,
     6506,
     5751,
     8927,
     2816,
     7348,
     10471,
     1648]

for i in a:
    if i not in b:
        print(i)
