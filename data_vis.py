import pandas as pd
import numpy as np

# 1. Load the data
df = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# print(df["subscribe"].value_counts())
# print(df.describe())

# 2. Data Preprocessing
# import matplotlib.pyplot as plt

# import seaborn as sns

# bins = [0, 143, 353, 1873, 5149]
# df1 = df[df["subscribe"] == "yes"]
# binning = pd.cut(df1["duration"], bins, right=False)
# time = pd.value_counts(binning)
# # 可视化
# time = time.sort_index()
# fig = plt.figure(figsize=(6, 2), dpi=120)
# sns.barplot(time.index, time, color="royalblue")
# x = np.arange(len(time))
# y = time.values
# for x_loc, jobs in zip(x, y):
#     plt.text(
#         x_loc,
#         jobs + 2,
#         "{:.1f}%".format(jobs / sum(time) * 100),
#         ha="center",
#         va="bottom",
#         fontsize=8,
#     )
# plt.xticks(fontsize=8)
# plt.yticks([])
# plt.ylabel("")
# plt.title("duration_yes", size=8)
# sns.despine(left=True)
# plt.show()
