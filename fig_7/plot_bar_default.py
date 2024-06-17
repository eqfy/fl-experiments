import numpy as np
import matplotlib.pyplot as plt


N = 4
# FedAvg, STC, APF, FedDC

downstreamMean = [39.3, 27.46, 19.65, 11.14]
upstreamMean = [66.8, 24.98, 18.4, 12.67]
computationMean = [6.75, 8.16, 8.43, 8.85]


downstreamStd = (7.2, 1.36, 3.4, 1.2) 
upstreamStd = (6.5, 1.84, 7.9, 1.5)
computationStd = (0.5, 0.42, 0.36, 0.46)


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
plt.xticks(ind, ("FedAvg", "STC", "APF", "GlueFL"))
plt.yticks(np.arange(0, 121, 30))
plt.legend((p1[0], p2[0], p3[0]), ('Download Time', 'Upload Time', 'Computation Time'), prop=font2)
ax1.grid(axis="y", linestyle="-.", zorder=0)


plt.xticks(fontsize=15, fontname="Times New Roman")
plt.yticks(fontsize=15)
plt.subplots_adjust(left=0.15, bottom=None, right=0.95, top=None, wspace=None, hspace=None)

ax1.set_axisbelow(True)

# plt.legend()
plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)

# plt.show()
plt.savefig("figs/bar_default.pdf")
plt.savefig("figs/bar_default.png")