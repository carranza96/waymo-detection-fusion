import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.colors as mcolors
from matplotlib import cm

plt.style.use(['science', 'no-latex', 'grid', "ieee", 'muted'])

# %%

df = pd.read_csv("results_more.csv", sep=",", index_col=None)
# df = df[~df["Detector"].str.contains("MobileNet")]

# %%

markers = ["o", "s", "^", "p", "*"]
feat_extractors = list(set(df["Feature extractor"].values))
feat_extractors = ['ResNet50', 'ResNet101', 'Res2Net101', 'ResNeXt101', 'ResNet152', 'DarkNet53', 'MobileNet', 'MobileNetV2']
n = 0
# colors = cm.get_cmap('cividis', len(feat_extractors)+n)
# colors = colors(range(len(feat_extractors)+n))[int(n/2):len(feat_extractors)+n-int(n/2-1)]
# colors = colors(range(len(feat_extractors)+n))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
_, ax = plt.subplots(1, 1, figsize=(5.5, 2.5))
ax2 = ax.twiny()

df["Name"] = df["Detector"] + " " + df["Feature extractor"]
for detector in df.Name.unique():
    x = df[df.Name == detector]["Inference time"]
    y = df[df.Name == detector]["mAP"]
    fe = df[df.Name == detector]["Feature extractor"].values[0]
    m = markers[df[df.Name == detector].Style.values[0] - 1]
    color = colors[feat_extractors.index(fe)]
    ax.plot(x, y, m + "-", label=detector, linewidth=0.2, color=color)

ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([33.333, 100])
ax2.set_xticklabels(["30 fps", "10 fps"])
ax2.grid(color="b", )

ax.legend(bbox_to_anchor=(1., 1.0325), ncol=1)
ax.set_ylabel("mAP")
ax.set_xlabel("Inference time (ms)")
plt.savefig("results_final.png", dpi=2000)
plt.show()

# %%

# res = "HIGH"
# object_types = ["Vehicle", "Pedestrian", "Cyclist"]
# markers = ["o", "s", "^", "p", "*"]
# _, ax = plt.subplots(1, len(object_types), figsize=(8.5, 2.5), sharex=True, sharey=True)
#
# for i, ot in enumerate(object_types):
#     rows = df[df["Resolution"] == res]
#     ax[i].set_title(ot)
#     for idx, row in rows.iterrows():
#         x = row["Inference time"]
#         y = row[ot]
#         l = row["Name"]
#         s = row["Style"] - 1
#         fe = row["Feature extractor"]
#         color = colors[feat_extractors.index(fe)]
#
#         ax[i].scatter(x, y, marker=markers[s], label=l, color=color)
#
# ax[0].set_ylabel("mAP")
# ax[1].set_xlabel("Inference time (ms)")
#
# ax[1].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
# plt.savefig("results_by_object.png", dpi=1000)
# plt.show()
