def find_min_downfactor(compute, downstream, model_sizes=model_sizes_c20, compress_rate=.2, upfactor=.2, addfactor=.1):
    single_upload_size = compress_rate * full_model_size
    for skipped_round in range(1, 19):
        # skipped_round == 1 means no prefetching at all
        total_fetch_amount = model_sizes[skipped_round]
        if skipped_round > 0:
            total_fetch_amount += model_sizes[1] * (skipped_round - 1)

        addition_time = compute * addfactor

        # prefactor is the min percentage of downstream that we need to use to achieve speedup
        prefactor = (model_sizes[skipped_round] + model_sizes[1] * (skipped_round - 1)) / \
            (skipped_round * (model_sizes[1] +
                              single_upload_size / upfactor + downstream * compute) - downstream * addition_time * (skipped_round - 1.))

        regular_time = skipped_round * \
            (model_sizes[1] / downstream + compute +
             single_upload_size / (downstream * upfactor))

        prefetch_time = model_sizes[skipped_round] / (downstream * prefactor) + \
            (model_sizes[1] / (downstream * prefactor) +
             addition_time) * (skipped_round - 1.)

        print(f"Prefetch {skipped_round} rounds with compute time {compute}s and downstream {downstream}MB/s:\n\
        \ttotal downstream: {total_fetch_amount}MB\n\
        \tminimal downstream factor: {prefactor}\n\
        \twith reg time {regular_time}s and prefetch time {prefetch_time}s")
