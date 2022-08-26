import numpy as np
import matplotlib.pyplot as plt


N = 4
# FedAvg, STC, APF, FedDC

downstreamMean = [0.95, 0.8, 0.782, 0.325]
upstreamMean = [1.66, 0.39, 0.733, 0.38]
computationMean = [12.01, 11.95, 12.04, 11.76]


downstreamStd = (0.046, 0.0765, 0.0277, 0.0605) 
upstreamStd = (0.018, 0.0048, 0.229, 0.0026)
computationStd = (0.081, 0.0776, 0.0869, 0.181)


upstreamMean = [(computationMean[i]) + upstreamMean[i] for i in range(4) ]
downstreamMean = [(upstreamMean[i]) + downstreamMean[i] for i in range(4) ]
print(downstreamMean)

ind = np.arange(N)    # the x locations for the groups
width = 0.8       # the width of the bars: can also be len(x) sequence

fig, ax1 = plt.subplots()
fig.set_size_inches(6, 3.6)

p1 = ax1.bar(ind, downstreamMean, width, color="grey", zorder=1, edgecolor="black", hatch="/")
p2 = ax1.bar(ind, upstreamMean, width, color="darkred", zorder=2, edgecolor="black")
p3 = ax1.bar(ind, computationMean, width, zorder=3, edgecolor="black", hatch="//")


label_size = 16
font2 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': label_size,
}

plt.ylabel('Training Time / Round (s)', font2)
# plt.title('')
plt.xticks(ind, ("FedAvg", "STC", "APF", "FedDC"))
plt.yticks(np.arange(0, 26, 5))
plt.legend((p1[0], p2[0], p3[0]), ('Download Time', 'Upload Time', 'Computation Time'), prop=font2)
ax1.grid(axis="y", linestyle="-.", zorder=0)


plt.xticks(fontsize=15, fontname="Times New Roman")
plt.yticks(fontsize=15)
plt.subplots_adjust(left=0.15, bottom=None, right=0.95, top=None, wspace=None, hspace=None)

ax1.set_axisbelow(True)

plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
# plt.legend()
# plt.show()
plt.savefig("figs/bar_gcp.pdf")