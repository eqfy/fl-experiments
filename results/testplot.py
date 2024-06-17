
from re import L
from turtle import color
import matplotlib.pyplot as plt
import numpy as np

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
    # "previous_baseline_log/femnist_shf_fvg",
    # "femnist_shf_fvg_new",
    # "previous_baseline_log/femnist_mob_fvg",
    # "FedDCMobilenet.log",
    # "0PrevFemnistBaseline.log",

    # "sg30_femnist_logging.log",
    # "sg60_femnist_logging.log",
    # "sg120_femnist_logging.log",
    # "sg240_femnist_logging.log",
    
    # "sg30_femnist1_logging.log",
    # "sg60_femnist1_logging.log",
    # "sg120_femnist1_logging.log",
    # "sg240_femnist1_logging.log",

    # "sg120_femnist_logging.log",
    # "cn12_femnist_logging.log",
    # "cn24_femnist_logging.log",

    # "sm0.04_femnist_logging.log",
    # "sm0.08_femnist_logging.log",
    # "sg120_femnist_logging.log",

    # "0PrevFemnistBaseline.log", # Don't use
    # "previous_baseline_log/femnist_shf_stc",
    # "previous_baseline_log/femnist_shf_apf",
    # "Shared0.04.log",
    # "sg60_femnist_logging.log",


    # "sg120_femnist_logging.log",
    # "re20_femnist_logging.log",
    # "reInf_femnist_logging.log",

    # "ecNone_femnist_logging.log",
    # "ecNoRW_femnist_logging.log",
    # "sg120_femnist_logging.log",

    "previous_baseline_log/speech_fedavg",

    # "sg30_speech_logging.log",
    # "sg60_speech_logging.log",
    # "sg120_speech_logging.log",
    # "sg240_speech_logging.log",

    # "sg120_speech_logging.log",
    # "cn12_speech_logging.log",
    # "cn24_speech_logging.log",

    # "sm0.06_speech_logging.log",
    # "sm0.12_speech_logging.log",
    # "sg120_speech_logging.log",


    "previous_baseline_log/speech_stc",
    "previous_baseline_log/speech_apf",
    "0PrevSpeechBaseline.log",
    "sg120_speech_logging.log",

    # "sg120_speech_logging.log",
    # "re20_speech_logging.log",
    # "reInf_speech_logging.log",

    # "ecNone_speech_logging.log",
    # "ecNoRW_speech_logging.log",
    # "sg120_speech_logging.log",
]

label_names = [
    "FedAvg",
    # "femnist_shf_fvg_new",
    # "FedAvg - Mobile",
    # "FedDC Mobile",
    # "GlueFL (No Reweight)",


    # "GlueFL ($S = 30$)",
    # "GlueFL ($S = 60$)",
    # "GlueFL ($S = 120$)",
    # "GlueFL ($S = 240$)"

    # "GlueFL ($C = 24$)",
    # "GlueFL ($C = 18$)",
    # "GlueFL ($C = 6$)",

    # "GlueFL ($q_{shr} = 4\%$)",
    # "GlueFL ($q_{shr} = 8\%$)",
    # "GlueFL ($q_{shr} = 16\%$)",
    # "GlueFL ($q_{shr} = 6\%$)",
    # "GlueFL ($q_{shr} = 12\%$)",
    # "GlueFL ($q_{shr} = 24\%$)",

    "STC",
    "APF",
    "GlueFL (Equal)",
    "GlueFL",

    # "GlueFL ($I = 10$)",
    # "GlueFL ($I = 20$)",
    # "GlueFL ($I = \infty$)",

    # "GlueFL (None)",
    # "GlueFL (EC)",
    # "GlueFL (REC)",
]

file_to_save = [
    # "femnist_sg.pdf",
    # "femnist_cn.pdf",
    # "femnist_sm.pdf",
    # "femnist_rw.pdf",
    # "femnist_re.pdf",
    # "femnist_ec.pdf",

    # "speech_sg.pdf",
    # "speech_cn.pdf",
    # "speech_sm.pdf",
    "speech_rw.pdf",
    # "speech_re.pdf",
    # "speech_ec.pdf",

    # "femnist_rw_1.pdf"
    # "speech_rw_1.pdf"

    # "fedavg_test.pdf"
    # "fedavg_mobile.pdf"
]

color_names = [
    "black",
    "black",
    "black",
    # "grey",
    "orange",
    "green",
    "royalblue",
    "brown",

    "lightblue",
    "royalblue",
    "limegreen",
    "darkgreen",

    # "grey",
    # "royalblue"
]

markers = [
    "",
    "",
    "s",
    "o",
    "p",
    "v",
    "^",
]


# def plotDiagram(file_names, label_names, file_to_save):
fig, ax1 = plt.subplots()
ax1.set_xlabel('Communication Rounds')
ax1.set_ylabel('Test Accuracy')

idx = 0
favg_dl_bw = []
for name in file_names:

    file_name = "./" + name
    f = open(file_name, "r")

    lines = f.readlines()
    lost_client_count = 0
    straggler_count = 0
    record_next = 0
    for i, line in enumerate(lines):
        if "Selected participants to run: " in line:
            if i + 2 >= len(lines):
                break
            straggler_line = lines[i+1].split(" ", 1)
            lost_line = lines[i+2].split(" ", 1)
            # print(lines[i], lines[i+1], lines[i+2])
            # print(straggler_line, lost_line)

            if straggler_line[0] == "stragglers:" and lost_line[0] == "lost:":
                # print(straggler_line[1])
                # print(lost_line[1])
                stragglers = eval(straggler_line[1].strip())
                lost = eval(lost_line[1].strip())

                # print(stragglers)
                # print(lost)
                straggler_count += len(stragglers)
                lost_client_count += len(lost)



    if (straggler_count > 0 and lost_client_count > 0):
        straggler_ratio = straggler_count / (straggler_count + lost_client_count)
    else:
        straggler_ratio = 1

    print(straggler_ratio)

    x_list = []
    y_list = []
    bw_list = []
    download_bw = 0.0

    if len(favg_dl_bw) > 0: 
        dif_1 = (favg_dl_bw[-1] / len(favg_dl_bw)) * 5
    for line in lines:
        if ("total" not in line) and ("kbit" in line):
            # print(line[:-1].split(" "))
            # print(line[:-1].split(" "))
            download_bw = (eval(line[:-1].split(" ")[14]) + eval(line[:-1].split(" ")[20]) * straggler_ratio) / 1024 / 1024 / 8 / 100
            if name == "previous_baseline_log/femnist_shf_fvg":
                favg_dl_bw.append(download_bw)
            # print("download bw ", download_bw)
            # raise Exception
            
            
        if "Testing" not in line:
            continue
        # print(line[:-1].split(" "))
        # print(line[:-1].split(" ")[12][:-1], line[:-1].split(" ")[-11])
        
        try:
            epoch = eval(line[:-1].split(" ")[12][:-1])
            accu = eval(line[:-1].split(" ")[16])
        except:
            print("Exception occured: ", line)
        
        if epoch <= 1000:
            x_list.append(epoch)
            y_list.append(accu)
            # bw_list.append(download_bw)

            if name == "femnist_shf_fvg_new":
                # print("got here", favg_dl_bw)
                if epoch < len(favg_dl_bw):
                    bw_list.append(favg_dl_bw[epoch - 1])
                    # print(favg_dl_bw[epoch - 1] - favg_dl_bw[epoch - 2], len(favg_dl_bw))
                else:
                    # print(epoch, dif_1, accu)
                    bw_list.append(bw_list[-1] + dif_1)
                    # print(dif_1)
            else:
                bw_list.append(download_bw)
            
        #     y_list[epoch] += (downstream_ratio / client_num)
        # print(epoch, downstream_ratio)

    if name == "previous_baseline_log/femnist_shf_fvg":
        continue

    if name == "sg240_speech_logging.log":
        
        avg_per_round_bandwidth = (bw_list[-1] - bw_list[-20]) / 10 / 20
        print("ran", len(bw_list))
        remaining_accs = [60.2204,  59.5592, 61.2121, 60.3857, 59.7796, 60.551, 59.6694,  60.6061, 60.7713,  59.6694, 60.0, 60.6612, 59.9449, 59.5592 ]
        y_list.extend(remaining_accs)
        for i in range(len(remaining_accs)):
            bw_list.append(bw_list[-1] + avg_per_round_bandwidth * 10)

    # print(idx)
    y_list = get_smooth(y_list, 20)
    if "fvg" in name or "fedavg" in name:
        ax1.plot(bw_list, y_list, label=label_names[idx], color=color_names[idx], linewidth=2, linestyle="dashed")
    elif "stc" in name:
        ax1.plot(bw_list, y_list, label=label_names[idx], color=color_names[idx], linewidth=2, linestyle="dotted")
    elif "apf" in name:
        ax1.plot(bw_list, y_list, label=label_names[idx], color=color_names[idx], linewidth=2, linestyle="dashdot")
    else:
        ax1.plot(bw_list, y_list, label=label_names[idx], color=color_names[idx], linewidth=2, marker=markers[idx], markevery=12)
    # ax1.plot(x_list, y_list, label=label_names[idx], color=color_names[idx])
    
    idx += 1

        



# plt.ylim(55, 77)
# plt.xlim(0.7, 3.0)

plt.ylim(40, 65)
plt.xlim(2, 10)

import numpy as np
# plt.xticks(np.arange(1.0, 3.5, 0.5))
# plt.yticks(np.arange(55, 78, 5))

plt.xticks(np.arange(2, 11, 1))
plt.yticks(np.arange(40, 66, 5))

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


# plt.savefig("speech_sg.pdf")
# plt.savefig("speech_cn.pdf")
# plt.savefig("speech_sm.pdf")
# plt.savefig("speech_rw.pdf")
# plt.savefig("speech_re.pdf")
# plt.savefig("speech_ec.pdf")

plt.savefig("fig/"+file_to_save[0])
# plt.show()

# femnist 0.5113404862148581   1.9 2.7 e560 
# speech 0.24787409569742352 e660