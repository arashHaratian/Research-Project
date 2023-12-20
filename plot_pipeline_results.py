import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 11.25)
plt.rcParams["font.size"] = (15)

results = pd.read_csv("pipeline_results.txt", sep=";", header=None, skiprows=1)
results = results.values


results_numeric = []
for res in results:
    results_numeric.append([float(x) for x in re.findall("\d+\.\d+[e\-\d+]*", res[0])])

results_numeric = np.array(results_numeric)

target_rankme_vicreg_single = results_numeric[:5, 0]
target_rankme_vicreg_multi = results_numeric[5:10, 0]
target_rankme_dino_single = results_numeric[10:15, 0]
target_rankme_dino_multi = results_numeric[15:20, 0]

nontarget_rankme_vicreg_single = results_numeric[:5, 1]
nontarget_rankme_vicreg_multi = results_numeric[5:10, 1]
nontarget_rankme_dino_single = results_numeric[10:15, 1]
nontarget_rankme_dino_multi = results_numeric[15:20, 1]

unbiased_rankme_vicreg_single = results_numeric[:5, 2]
unbiased_rankme_vicreg_multi = results_numeric[5:10, 2]
unbiased_rankme_dino_single = results_numeric[10:15, 2]
unbiased_rankme_dino_multi = results_numeric[15:20, 2]


target_is_vicreg_single = results_numeric[:5, 3]
target_is_vicreg_single_std = results_numeric[:5, 4]
target_is_vicreg_multi = results_numeric[5:10, 3]
target_is_vicreg_multi_std = results_numeric[5:10, 4]
target_is_dino_single = results_numeric[10:15, 3]
target_is_dino_single_std = results_numeric[10:15, 4]
target_is_dino_multi = results_numeric[15:20, 3]
target_is_dino_multi_std = results_numeric[15:20, 4]

nontarget_is_vicreg_single = results_numeric[:5, 5]
nontarget_is_vicreg_single_std = results_numeric[:5, 6]
nontarget_is_vicreg_multi = results_numeric[5:10, 5]
nontarget_is_vicreg_multi_std = results_numeric[5:10, 6]
nontarget_is_dino_single = results_numeric[10:15, 5]
nontarget_is_dino_single_std = results_numeric[10:15, 6]
nontarget_is_dino_multi = results_numeric[15:20, 5]
nontarget_is_dino_multi_std = results_numeric[15:20, 6]

unbiased_is_vicreg_single = results_numeric[:5, 7]
unbiased_is_vicreg_single_std = results_numeric[:5, 8]
unbiased_is_vicreg_multi = results_numeric[5:10, 7]
unbiased_is_vicreg_multi_std = results_numeric[5:10, 8]
unbiased_is_dino_single = results_numeric[10:15, 7]
unbiased_is_dino_single_std = results_numeric[10:15, 8]
unbiased_is_dino_multi = results_numeric[15:20, 7]
unbiased_is_dino_multi_std = results_numeric[15:20, 8]


target_fid_vicreg_single = results_numeric[:5, 9]
target_fid_vicreg_multi = results_numeric[5:10, 9]
target_fid_dino_single = results_numeric[10:15, 9]
target_fid_dino_multi = results_numeric[15:20, 9]

nontarget_fid_vicreg_single = results_numeric[:5, 10]
nontarget_fid_vicreg_multi = results_numeric[5:10, 10]
nontarget_fid_dino_single = results_numeric[10:15, 10]
nontarget_fid_dino_multi = results_numeric[15:20, 10]

unbiased_fid_vicreg_single = results_numeric[:5, 11]
unbiased_fid_vicreg_multi = results_numeric[5:10, 11]
unbiased_fid_dino_single = results_numeric[10:15, 11]
unbiased_fid_dino_multi = results_numeric[15:20, 11]




target_rankme_vicreg_single_mean = np.mean(target_rankme_vicreg_single)
target_rankme_vicreg_single_std = np.std(target_rankme_vicreg_single)
target_rankme_vicreg_multi_mean = np.mean(target_rankme_vicreg_multi)
target_rankme_vicreg_multi_std = np.std(target_rankme_vicreg_multi)
target_rankme_dino_single_mean = np.mean(target_rankme_dino_single)
target_rankme_dino_single_std = np.std(target_rankme_dino_single)
target_rankme_dino_multi_mean = np.mean(target_rankme_dino_multi)
target_rankme_dino_multi_std = np.std(target_rankme_dino_multi)

nontarget_rankme_vicreg_single_mean = np.mean(nontarget_rankme_vicreg_single)
nontarget_rankme_vicreg_single_std = np.std(nontarget_rankme_vicreg_single)
nontarget_rankme_vicreg_multi_mean = np.mean(nontarget_rankme_vicreg_multi)
nontarget_rankme_vicreg_multi_std = np.std(nontarget_rankme_vicreg_multi)
nontarget_rankme_dino_single_mean = np.mean(nontarget_rankme_dino_single)
nontarget_rankme_dino_single_std = np.std(nontarget_rankme_dino_single)
nontarget_rankme_dino_multi_mean = np.mean(nontarget_rankme_dino_multi)
nontarget_rankme_dino_multi_std = np.std(nontarget_rankme_dino_multi)

unbiased_rankme_vicreg_single_mean = np.mean(unbiased_rankme_vicreg_single)
unbiased_rankme_vicreg_single_std = np.std(unbiased_rankme_vicreg_single)
unbiased_rankme_vicreg_multi_mean = np.mean(unbiased_rankme_vicreg_multi)
unbiased_rankme_vicreg_multi_std = np.std(unbiased_rankme_vicreg_multi)
unbiased_rankme_dino_single_mean = np.mean(unbiased_rankme_dino_single)
unbiased_rankme_dino_single_std = np.std(unbiased_rankme_dino_single)
unbiased_rankme_dino_multi_mean = np.mean(unbiased_rankme_dino_multi)
unbiased_rankme_dino_multi_std = np.std(unbiased_rankme_dino_multi)




########################################################################################################################
#                                               PLOTTING VICREG OF RANKME VS IS
########################################################################################################################
fig, (axs1, axs2) = plt.subplots(1, 2)

axs1.plot(target_rankme_vicreg_single, color="blue", label = "target (gender = 1)")
axs1.errorbar(5, target_rankme_vicreg_single_mean, yerr = target_rankme_vicreg_single_std, fmt ='.', label = "average over runs and its Std.")
axs1.plot(nontarget_rankme_vicreg_single, color="red", label = "nontarget (gender = 0)")
axs1.errorbar(5, nontarget_rankme_vicreg_single_mean, yerr = nontarget_rankme_vicreg_single_std, fmt ='.')
axs1.plot(unbiased_rankme_vicreg_single, color="green", label = "unbiased")
axs1.errorbar(5, unbiased_rankme_vicreg_single_mean, yerr = unbiased_rankme_vicreg_single_std, fmt ='.')
axs1.set_title("RankMe Scores")
axs1.set(xlabel = "Number of run", ylabel = "RankMe score")

axs2.plot(target_is_vicreg_single, color="blue")
axs2.plot(target_is_vicreg_single + target_is_vicreg_single_std, color="blue", linestyle = "--", label = "Std. at each run")
axs2.plot(target_is_vicreg_single - target_is_vicreg_single_std, color="blue", linestyle = "--")
axs2.plot(nontarget_is_vicreg_single, color="red")
axs2.plot(nontarget_is_vicreg_single + nontarget_is_vicreg_single_std, color="red", linestyle = "--")
axs2.plot(nontarget_is_vicreg_single - nontarget_is_vicreg_single_std, color="red", linestyle = "--")
axs2.plot(unbiased_is_vicreg_single, color="green")
axs2.plot(unbiased_is_vicreg_single + unbiased_is_vicreg_single_std, color="green", linestyle = "--")
axs2.plot(unbiased_is_vicreg_single - unbiased_is_vicreg_single_std, color="green", linestyle = "--")
axs2.set_title("Inception Scores")
axs2.set(xlabel = "Number of run", ylabel = "Inception score")

fig.legend()
fig.suptitle('RankMe scores and Inception scores for single target')
# fig.show()
fig.savefig("plots/vicreg_single")

########################################################################################################################

fig, (axs1, axs2) = plt.subplots(1, 2)
axs1.plot(target_rankme_vicreg_multi, color="blue", label = "target (gender = 1)")
axs1.errorbar(5, target_rankme_vicreg_multi_mean, yerr = target_rankme_vicreg_multi_std, fmt ='.', label = "average over runs and its Std.")
axs1.plot(nontarget_rankme_vicreg_multi, color="red", label = "nontarget (gender = 0)")
axs1.errorbar(5, nontarget_rankme_vicreg_multi_mean, yerr = nontarget_rankme_vicreg_multi_std, fmt ='.')
axs1.plot(unbiased_rankme_vicreg_multi, color="green", label = "unbiased")
axs1.errorbar(5, unbiased_rankme_vicreg_multi_mean, yerr = unbiased_rankme_vicreg_multi_std, fmt ='.')
axs1.set_title("RankMe Scores")
axs1.set(xlabel = "Number of run", ylabel = "RankMe score")

axs2.plot(target_is_vicreg_multi, color="blue")
axs2.plot(target_is_vicreg_multi + target_is_vicreg_multi_std, color="blue", linestyle = "--", label = "Std. at each run")
axs2.plot(target_is_vicreg_multi - target_is_vicreg_multi_std, color="blue", linestyle = "--")
axs2.plot(nontarget_is_vicreg_multi, color="red")
axs2.plot(nontarget_is_vicreg_multi + nontarget_is_vicreg_multi_std, color="red", linestyle = "--")
axs2.plot(nontarget_is_vicreg_multi - nontarget_is_vicreg_multi_std, color="red", linestyle = "--")
axs2.plot(unbiased_is_vicreg_multi, color="green")
axs2.plot(unbiased_is_vicreg_multi + unbiased_is_vicreg_multi_std, color="green", linestyle = "--")
axs2.plot(unbiased_is_vicreg_multi - unbiased_is_vicreg_multi_std, color="green", linestyle = "--")
axs2.set_title("Inception Scores")
axs2.set(xlabel = "Number of run", ylabel = "Inception score")

fig.legend()
fig.suptitle('RankMe scores and Inception scores for multi-target')
# fig.show()
fig.savefig("plots/vicreg_multi")

########################################################################################################################
#                                               PLOTTING DINO OF RANKME VS IS
########################################################################################################################


fig, (axs1, axs2) = plt.subplots(1, 2)

axs1.plot(target_rankme_dino_single, color="blue", label = "target (gender = 1)")
axs1.errorbar(5, target_rankme_dino_single_mean, yerr = target_rankme_dino_single_std, fmt ='.', label = "average over runs and its Std.")
axs1.plot(nontarget_rankme_dino_single, color="red", label = "nontarget (gender = 0)")
axs1.errorbar(5, nontarget_rankme_dino_single_mean, yerr = nontarget_rankme_dino_single_std, fmt ='.')
axs1.plot(unbiased_rankme_dino_single, color="green", label = "unbiased")
axs1.errorbar(5, unbiased_rankme_dino_single_mean, yerr = unbiased_rankme_dino_single_std, fmt ='.')
axs1.set_title("RankMe Scores")
axs1.set(xlabel = "Number of run", ylabel = "RankMe score")

axs2.plot(target_is_dino_single, color="blue")
axs2.plot(target_is_dino_single + target_is_dino_single_std, color="blue", linestyle = "--", label = "Std. at each run")
axs2.plot(target_is_dino_single - target_is_dino_single_std, color="blue", linestyle = "--")
axs2.plot(nontarget_is_dino_single, color="red")
axs2.plot(nontarget_is_dino_single + nontarget_is_dino_single_std, color="red", linestyle = "--")
axs2.plot(nontarget_is_dino_single - nontarget_is_dino_single_std, color="red", linestyle = "--")
axs2.plot(unbiased_is_dino_single, color="green")
axs2.plot(unbiased_is_dino_single + unbiased_is_dino_single_std, color="green", linestyle = "--")
axs2.plot(unbiased_is_dino_single - unbiased_is_dino_single_std, color="green", linestyle = "--")
axs2.set_title("Inception Scores")
axs2.set(xlabel = "Number of run", ylabel = "Inception score")

fig.legend()
fig.suptitle('RankMe scores and Inception scores for single target')
# fig.show()
fig.savefig("plots/dino_single")

########################################################################################################################


fig, (axs1, axs2) = plt.subplots(1, 2)
axs1.plot(target_rankme_dino_multi, color="blue", label = "target (gender = 1)")
axs1.errorbar(5, target_rankme_dino_multi_mean, yerr = target_rankme_dino_multi_std, fmt ='.', label = "average over runs and its Std.")
axs1.plot(nontarget_rankme_dino_multi, color="red", label = "nontarget (gender = 0)")
axs1.errorbar(5, nontarget_rankme_dino_multi_mean, yerr = nontarget_rankme_dino_multi_std, fmt ='.')
axs1.plot(unbiased_rankme_dino_multi, color="green", label = "unbiased")
axs1.errorbar(5, unbiased_rankme_dino_multi_mean, yerr = unbiased_rankme_dino_multi_std, fmt ='.')
axs1.set_title("RankMe Scores")
axs1.set(xlabel = "Number of run", ylabel = "RankMe score")

axs2.plot(target_is_dino_multi, color="blue")
axs2.plot(target_is_dino_multi + target_is_dino_multi_std, color="blue", linestyle = "--", label = "Std. at each run")
axs2.plot(target_is_dino_multi - target_is_dino_multi_std, color="blue", linestyle = "--")
axs2.plot(nontarget_is_dino_multi, color="red")
axs2.plot(nontarget_is_dino_multi + nontarget_is_dino_multi_std, color="red", linestyle = "--")
axs2.plot(nontarget_is_dino_multi - nontarget_is_dino_multi_std, color="red", linestyle = "--")
axs2.plot(unbiased_is_dino_multi, color="green")
axs2.plot(unbiased_is_dino_multi + unbiased_is_dino_multi_std, color="green", linestyle = "--")
axs2.plot(unbiased_is_dino_multi - unbiased_is_dino_multi_std, color="green", linestyle = "--")
axs2.set_title("Inception Scores")
axs2.set(xlabel = "Number of run", ylabel = "Inception score")

fig.legend()
fig.suptitle('RankMe scores and Inception scores for multi-target')
# fig.show()
fig.savefig("plots/dino_multi")