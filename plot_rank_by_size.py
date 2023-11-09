import torch
from pipeline import load_data, load_model
from RankMe import RankMe
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 11.25)

model = load_model("vicreg")

sample_size = 10240

data_loader = load_data(sample_size)
data = next(iter(data_loader))[0]

rankme_target = RankMe(model, 256)

res = rankme_target.evaluate_with_size(data, torch.arange(0, sample_size, 256), True)

del(data)

plt.plot(torch.arange(0, sample_size, 256), res)
plt.xlabel("Number of data points")
plt.ylabel("Rank")
plt.title("Re-producing the Figure S11 from original data")
# plt.show()
plt.savefig("plots/replot_S11")