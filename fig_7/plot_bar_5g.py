import numpy as np
import matplotlib.pyplot as plt


N = 4
# FedAvg, STC, APF, FedDC

downstreamMean = [1.8, 1.54, 1.7, 0.94]
upstreamMean = [2.37, 0.55, 1.1, 0.56]
computationMean = [11.15, 11.24, 11.0, 10.1]


downstreamStd = (0.27, 0.3, 0.33, 0.35) 
upstreamStd = (0.43, 0.02, 0.372, 0.023)
computationStd = (0.546, 0.384, 0.429, 0.505)


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
# plt.savefig("figs/bar_5g.eps")
plt.savefig("figs/bar_5g.pdf")