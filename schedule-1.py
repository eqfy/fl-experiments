"""
Scheduler for prefetching by 1 round
"""

# Client selector
# the clients selected for prefetching
prefetch_clients = []  # maybe should be a deque

# the clients that no longer have overhead when joining training
# sticky clients should be first part of prepared clients
prepared_clients = []

# On conclusion of last training round

# update new prepared_clients
prepared_idx = 0
last_sticky_id = 10
used_prefetch = []
for i, client in enumerate(prefetch_clients):
    if isValid(client):
        prepared_clients[last_sticky_id + prepared_idx] = client
        prepared_idx += 1

# prepared_new_clients = True
# prefetch_idx = 0
# while prepared_new_clients:
#     client = prefetch_clients[prefetch_idx]


# find new prefetch_clients
# ignore clients in prepared_clients, keep clients in prefetch_clients
prefetch_clients = random_sample(prepared_clients, prefetch_clients)

# start new round with prepared_clients

# start new round with prefetch_clients
