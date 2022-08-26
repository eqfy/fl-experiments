import numpy as np
import matplotlib.pyplot as plt


N = 4
# FedAvg, STC, APF, FedDC
computationMean = [126, 127, 131.1, 130]
downstreamMean = [111, 7.3, 9.5, 13]
upstreamMean = [179.45, 120.1, 123.3, 120.5]

computationStd = (4, 5, 5.4, 4)
downstreamStd = (0, 3.3 * 0.5, 4 * 0.5, 5 * 0.5) 
upstreamStd = (11, 20, 16, 15)


upstreamMean = [(computationMean[i]) + upstreamMean[i] for i in range(4) ]
downstreamMean = [(upstreamMean[i]) + downstreamMean[i] for i in range(4) ]
print(downstreamMean)

ind = np.arange(N)    # the x locations for the groups
width = 0.8       # the width of the bars: can also be len(x) sequence

fig, ax1 = plt.subplots()

p3 = ax1.bar(ind, computationMean, width, yerr=computationStd, zorder=3, edgecolor="black")
p2 = ax1.bar(ind, upstreamMean, width, yerr=upstreamStd, color="darkred", zorder=2, edgecolor="black")
p1 = ax1.bar(ind, downstreamMean, width, yerr=downstreamStd, color="grey", zorder=1, edgecolor="black")


label_size = 18
font2 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': label_size,
}

plt.ylabel('Training Time / Epoch (s)', font2)
# plt.title('')
plt.xticks(ind, ("FedAvg", "STC", "APF", "FedDC"))
plt.yticks(np.arange(0, 500, 50))
plt.legend((p1[0], p2[0], p3[0]), ('Download Time', 'Upload Time', 'Computation Time'), prop=font2)
ax1.grid(axis="y", linestyle="-.", zorder=0)


plt.xticks(fontsize=12.5, fontname="Times New Roman")
plt.yticks(fontsize=15)
plt.subplots_adjust(left=0.15, bottom=None, right=0.95, top=None, wspace=None, hspace=None)

ax1.set_axisbelow(True)

# plt.legend()
plt.show()