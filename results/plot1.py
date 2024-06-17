
from turtle import color
import matplotlib.pyplot as plt

def get_smooth(x_list, avg_cnt):
    x_avg_list = [0]

    for i in range(1, len(x_list)):
        idx = max(1, i - avg_cnt + 1)
        cnt = 0
        tot = 0
        for j in range(idx, i + 1):
            cnt += 1
            tot += x_list[j]
        x_avg_list.append(tot / cnt)

    return x_avg_list


# file_names = [
#     "FedAvg/femnist_fedavg.txt",
#     # "Sticky group size/30.log",
#     # "Sticky group size/60.log",
#     # "Sticky group size/120.log",
#     # "Sticky group size/240.log",
# ]

file_names = [
    "FedAvg/femnist_fedavg_acc.log",
    # "Sticky group size/30_acc.log",
    # "Sticky group size/60_acc.log",
    # "Sticky group size/120_acc.log",
    # "Sticky group size/240_acc.log",

    "Change ratio/change6_acc.log",
    "Change ratio/change12_acc.log",
    "Change ratio/change24_acc.log",
    
    # "Shared mask ratio/shared0.16_acc.log",
    # "Shared mask ratio/shared0.08_acc.log",
    # "Shared mask ratio/shared0.04_acc.log",

]

label_names = [
    "FedAvg",
    # "FedDC ($S = 30$)",
    # "FedDC ($S = 60$)",
    # "FedDC ($S = 120$)",
    # "FedDC ($S = 240$)",

    "FedDC ($C = 24$)",
    "FedDC ($C = 18$)",
    "FedDC ($C = 6$)",

    # "FedDC ($q_{shr} = 16\%$)",
    # "FedDC ($q_{shr} = 8\%$)",
    # "FedDC ($q_{shr} = 4\%$)",

]

color_names = [
    "black",
    # "grey",
    "orange",
    "green",
    "royalblue",
    "brown"

    # "lightblue",
    # "royalblue",
    # "limegreen",
    # "darkgreen",

    # "grey",
    # "royalblue"
]

# name = "03_01_Downstream/dr_w28_fvg_c01.out"

fig, ax1 = plt.subplots()
ax1.set_xlabel('Communication Rounds')
ax1.set_ylabel('Test Accuracy')

idx = 0
for name in file_names:

    file_name = "./hyperparameters/" + name
    f = open(file_name, "r")

    lines = f.readlines()
    
    x_list = []
    y_list = []
    bw_list = []
    download_bw = 0.0
    for line in lines:
        if ("total" not in line) and ("kbit" in line):
            # print(line[:-1].split(" "))
            # print(line[:-1].split(" "))
            download_bw = (eval(line[:-1].split(" ")[14]) + eval(line[:-1].split(" ")[20])) / 1024 / 1024 / 8 / 100
            
            
        if "Testing" not in line:
            continue
        # print(line[:-1].split(" "))
        # print(line[:-1].split(" ")[12][:-1], line[:-1].split(" ")[-11])
        
        epoch = eval(line[:-1].split(" ")[12][:-1])
        accu = eval(line[:-1].split(" ")[16])
        
        if epoch <= 1000:
            x_list.append(epoch)
            y_list.append(accu)
            bw_list.append(download_bw)
        #     y_list[epoch] += (downstream_ratio / client_num)
        # print(epoch, downstream_ratio)

    # print(idx)
    y_list = get_smooth(y_list, 20)
    if "fedavg" in name:
        ax1.plot(bw_list, y_list, label=label_names[idx], color=color_names[idx], linewidth=2, linestyle="dashed")
    else:
        ax1.plot(bw_list, y_list, label=label_names[idx], color=color_names[idx], linewidth=2)
    # ax1.plot(x_list, y_list, label=label_names[idx], color=color_names[idx])
    
    idx += 1



plt.ylim(50, 78)
plt.xlim(0, 4)
import numpy as np
plt.xticks(np.arange(0, 4, 0.5))
plt.yticks(np.arange(50, 85, 5))
# plt.xticks(np.arange(0, 200, 10))

label_size = 18
font2 = {
    'weight': 'normal',
    'size': label_size,
}
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90, wspace=None, hspace=None)

ax1.set_xlabel(r'Cumulative Downstream Bandwidth ($\times 10^2 GB$)', font2)
ax1.set_ylabel('Test Accuracy', font2)
ax1.tick_params(axis='x', labelsize= label_size)
ax1.tick_params(axis='y', labelsize= label_size)

plt.grid(linestyle="-.")
plt.legend(loc="lower right", fontsize=label_size)
plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
plt.show()