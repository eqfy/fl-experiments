import numpy as np



names = [
    '/Users/qfyan/eric/fl-experiments/fig_7/default/femnist_fedavg_acc.log',
    '/Users/qfyan/eric/fl-experiments/fig_7/default/stc_acc.log',
    '/Users/qfyan/eric/fl-experiments/fig_7/default/apf_acc.log',
    '/Users/qfyan/eric/fl-experiments/fig_7/default/feddc_60_acc.log',
    
    
    # '/Users/qfyan/eric/fl-experiments/fig_7/5G/5GFedAvg_logging.log',
    # '/Users/qfyan/eric/fl-experiments/fig_7/5G/5GSTC_logging.log',
    # '/Users/qfyan/eric/fl-experiments/fig_7/5G/5GAPF_logging.log',
    # '/Users/qfyan/eric/fl-experiments/fig_7/5G/5GFedDC_logging.log',

    # '/Users/qfyan/eric/fl-experiments/fig_7/GCP/GCP_FedAvg_logging.log',
    # '/Users/qfyan/eric/fl-experiments/fig_7/GCP/GCP_STC_logging.log',
    # '/Users/qfyan/eric/fl-experiments/fig_7/GCP/GCP_APF_logging.log',
    # '/Users/qfyan/eric/fl-experiments/fig_7/GCP/GCP_FedDC_logging.log',
]


for name in names:
    round_dls = []
    round_uls = []
    round_computes = []
    prev_total_dl, prev_total_ul, prev_total_compute = 0,0,0


    round_avg_dls = []
    round_avg_uls = []
    round_avg_computes = []

    with open(name) as log_f:
        for line in log_f:
            if not line.startswith("             "):
                continue
            splitted = line.split()
            if "total_dl:" in splitted:
                total_dl, total_ul, total_compute = float(splitted[1]), float(splitted[4]), float(splitted[7])
                round_dls.append(total_dl- prev_total_dl)
                round_uls.append(total_ul - prev_total_ul)
                round_computes.append(total_compute - prev_total_compute)
                prev_total_dl, prev_total_ul, prev_total_compute = total_dl, total_ul, total_compute
            if "avg_dl:" in splitted:
                round_avg_dls.append(float(splitted[1]))
                round_avg_uls.append(float(splitted[4]))
                round_avg_computes.append(float(splitted[7]))


    sigma_dl = np.std(round_dls[-10:])
    sigma_ul = np.std(round_uls[-10:])
    sigma_compute = np.std(round_computes[-10:])

    sigma_avg_dl = np.std(round_avg_dls)
    sigma_avg_ul = np.std(round_avg_uls)
    sigma_avg_compute = np.std(round_avg_computes)


    print(f"name: {name}\nAvg: dl: {np.average(round_dls)}, ul: {np.average(round_uls)}, compute: {np.average(round_computes)}\nStd: avg_dl: {sigma_avg_dl}, avg_ul: {sigma_avg_ul}, avg_compute: {sigma_avg_compute}\n5 Percentile: dl: {np.percentile(round_dls[-50:], 5)} ul: {np.percentile(round_uls[-50:], 5)} compute: {np.percentile(round_computes[-50:], 5)}\n95 Percentile: dl: {np.percentile(round_dls[-50:], 95)} ul: {np.percentile(round_uls[-50:], 95)} compute: {np.percentile(round_computes[-50:], 95)}")



    # \nStd: dl: {sigma_dl}, ul: {sigma_ul}, compute: {sigma_compute}
    # Min: dl: {np.min(round_dls[-10:])} ul: {np.min(round_uls[-10:])} compute: {np.min(round_computes[-10:])}\nMax: dl: {np.max(round_dls[-10:])} ul: {np.max(round_uls[-10:])} computes: {np.max(round_computes[-10:])}
