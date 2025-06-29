"""
SIMULATION SETUP
"""

# %%
import numpy as np
from matplotlib import pyplot as plt
import pickle
import cgt
from cgt import *

# %%
# Choose n
n = 7

# %%
# Directory for results
results_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/results/"
figures_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/figures/"
results_file_name = f"distribution_results_n{n}.pkl"


# %%
# Set up simulation results dictionary
sim_model_dicts = {
    "IE": {MODEL.all_inversions: 1},
    "IU": {MODEL.all_inversions_larger_less_likely: 1},
    "SIE": {MODEL.one_region_inversions: 1 / 2, MODEL.two_region_inversions: 1 / 2},
    "SIU": {MODEL.one_region_inversions: 2 / 3, MODEL.two_region_inversions: 1 / 3},
}


# %%
simulation_results = {
    "n": n,
    "models": {
        model_name: {
            "model_dict": model_dict,
            "min_dists": None,  # "min_dist" : set(genomes)
            "stepwise_probs": None,  # "genome" : "stepwise_probs"
            "MFPTs": None,  # "genome" : "MFPT"
            "MFPTs_per_min_dist": None,  # "min_dist" : "MFPTs"
            "mles": None,  # "genome" : "mle"
            "mles_per_min_dist": None,  # "min_dist" : "mles"
        }
        for model_name, model_dict in sim_model_dicts.items()
    },
}


# %%
# Set up framework and models
framework = cgt.GenomeFramework(n)
sim_models = {
    name: cgt.Model.named_model_with_relative_probs(framework, model_dict)
    for name, model_dict in sim_model_dicts.items()
}

# %%
# SELECT MODEL
MODEL_NAME = "SIE"
results_for_model = simulation_results["models"][MODEL_NAME]
model = sim_models[MODEL_NAME]


# %%
# Min distances
if results_for_model["min_dists"] is None:
    min_dists = {}
    stepwise_probs = cgt.distances.fast_step_probabilities(framework, model)
    for genome, probs in stepwise_probs.items():
        min_distance = np.nonzero(probs)[0][0]
        if min_distance not in min_dists:
            min_dists[min_distance] = set()
        min_dists[min_distance].add(genome)
    results_for_model["min_dists"] = min_dists
    results_for_model["stepwise_probs"] = stepwise_probs

# %%
# MFPTs per min dist
if results_for_model["MFPTs"] is None:
    MFPTs = cgt.distances.fast_MFPT(framework, model)
    results = {}
    for min_dist, genomes in min_dists.items():
        if min_dist != 0:
            results[min_dist] = []
            for genome in genomes:
                results[min_dist].append(MFPTs[genome])
    results = dict(sorted(results.items()))
    results_for_model["MFPTs"] = MFPTs
    results_for_model["MFPTs_per_min_dist"] = results

# %%
# Compute MLEs
if results_for_model["mles"] is None:
    double_cosets = framework.canonical_double_cosets(join_inverse_classes=True)
    total_cosets = len(double_cosets)
    computed_cosets = 0
    mles = {}
    all_mles = {}
    for min_dist, genomes in min_dists.items():
        if len(all_mles) >= framework.num_genomes() - 1:  # minus one for identity
            break
        if min_dist == 0:
            continue
        mles[min_dist] = []
        for genome in genomes:
            # % Complete
            print(
                f"{round((computed_cosets / total_cosets) * 100, 1)}% complete",
                end="\r",
            )
            if genome not in all_mles:
                mle = distances.mle(framework, model, genome)
                computed_cosets += 1
                for coset in double_cosets:
                    if genome in coset:
                        for g in coset:
                            all_mles[g] = mle
                            mles[min_dist].append(mle)
                        break
    mles = dict(sorted(mles.items()))
    results_for_model["mles"] = all_mles
    results_for_model["mles_per_min_dist"] = mles

# %%
# Save results as a pickle
with open(results_dir + results_file_name, "wb") as f:
    pickle.dump(simulation_results, f)

# %%
""" PLOTTING RESULTS """

# %%
# Load results
with open(results_dir + results_file_name, "rb") as f:
    simulation_results = pickle.load(f)

# %%
MODEL_NAME = "IE"

# %%
# Set up plotting
cgt.plotting.latex_figure_setup()
colors = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]

# Desaturated colours
model_colors = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]

PRINT_TITLES = False

# %%
# Plot proportion of NaNs
proportion_of_nans_per_min_dist = {}
for model_name in simulation_results["models"].keys():
    mles_no_nans = {}
    mles = simulation_results["models"][model_name]["mles_per_min_dist"]
    for min_dist, mle_list in mles.items():
        mle_vals = [mle for mle in mle_list if not np.isnan(mle)]
        if len(mle_vals) > 0:
            mles_no_nans[min_dist] = mle_vals
        else:
            mles_no_nans[min_dist] = []
    mles_no_nans = dict(sorted(mles_no_nans.items()))
    proportion_of_nans_per_min_dist[model_name] = {
        min_dist: (len(mles_no_nans[min_dist])) / len(mle_list)
        for min_dist, mle_list in mles.items()
    }

cgt.plotting.latex_figure_setup()
fig, axes = plt.subplots()
fig.set_size_inches(5, 2.45)
fig.tight_layout()
fig.subplots_adjust(right=0.95, bottom=0.2, top=0.98, left=0.11)
fig.set_dpi(300)
for i, model_name in enumerate(proportion_of_nans_per_min_dist.keys()):
    values = list(proportion_of_nans_per_min_dist[model_name].values())
    labels = list(proportion_of_nans_per_min_dist[model_name].keys())
    # plot clustered bar chart with offsets
    width = 0.175
    multiplier = i - len(proportion_of_nans_per_min_dist) / 2.9
    offset = width * multiplier
    plt.bar(
        [x + offset for x in range(1, len(labels) + 1)],
        values,
        width=width,
        label=model_name,
        color=model_colors[i],
        edgecolor="black",
        linewidth=0.5,
    )
plt.xticks(range(1, len(labels) + 1), labels)
plt.xlabel("Minimum Distance")
plt.ylabel("Proportion of genomes")
if PRINT_TITLES: plt.title(f"Proportion of genomes with an MLE, {n} regions", y=1.05)
# Remove border from legend
plt.legend(frameon=False)
plt.savefig(
    figures_dir + f"proportion_of_MLEs_per_min_dist_{n}_untitled.pdf"
)


# %%
# Plot MLE Boxplots
fig, axes = plt.subplots()
fig.set_size_inches(4, 3)
fig.tight_layout()
fig.set_dpi(300)
mles_no_nans = {}
mles = simulation_results["models"][MODEL_NAME]["mles_per_min_dist"]
for min_dist, mle_list in mles.items():
    mle_vals = [mle for mle in mle_list if not np.isnan(mle)]
    if len(mle_vals) > 0:
        mles_no_nans[min_dist] = mle_vals
    else:
        mles_no_nans[min_dist] = []
mles_no_nans = dict(sorted(mles_no_nans.items()))
mles_no_nans_without_empty = {
    key: val for key, val in mles_no_nans.items() if len(val) > 0
}
values = list(mles_no_nans_without_empty.values())
labels = list(mles_no_nans_without_empty.keys())
boxes = plt.boxplot(
    values,
    widths=0.5,
    patch_artist=True,
    flierprops=dict(marker="o", markersize=1, linewidth=0.1),
)
for b, box in enumerate(boxes["boxes"]):
    box.set(facecolor=colors[b], linewidth=1)
plt.xticks(range(1, len(labels) + 1), labels)
plt.xlabel("Minimum Distance")
plt.ylabel("MLE")
if PRINT_TITLES: plt.title(f"MLE per Min Distance for $M_{{{MODEL_NAME}}}$ on {n} regions", y=1.05)
plt.savefig(figures_dir + f"MLE_per_min_dist_{n}_{MODEL_NAME}_untitled.pdf", bbox_inches="tight")

#%%
# MLE boxplots as subplots
fig, axes = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(6, 2.7)
fig.tight_layout()
fig.subplots_adjust(wspace=0.09)
fig.set_dpi(300)
model_groupings = [["SIE", "SIU"], ["IE", "IU"]]
if MODEL_NAME in model_groupings[0]:
    model_groupings = model_groupings[0]
else:
    model_groupings = model_groupings[1]
for ax, model_name in zip(axes, model_groupings):
    mles_no_nans = {}
    mles = simulation_results["models"][model_name]["mles_per_min_dist"]
    for min_dist, mle_list in mles.items():
        mle_vals = [mle for mle in mle_list if not np.isnan(mle)]
        if len(mle_vals) > 0:
            mles_no_nans[min_dist] = mle_vals
        else:
            mles_no_nans[min_dist] = []
    mles_no_nans = dict(sorted(mles_no_nans.items()))
    mles_no_nans_without_empty = {
        key: val for key, val in mles_no_nans.items() if len(val) > 0
    }
    values = list(mles_no_nans_without_empty.values())
    labels = list(mles_no_nans_without_empty.keys())
    boxes = ax.boxplot(
        values,
        widths=0.5,
        patch_artist=True,
        flierprops=dict(marker="o", markersize=1, linewidth=0.1),
    )
    for b, box in enumerate(boxes["boxes"]):
        box.set(facecolor=colors[b], linewidth=1)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_title(f"$M_{{{model_name}}}$")
    
titlex = fig.supxlabel("Minimum Distance", x=0.5, y=-0.03)
ttiley = fig.supylabel("MLE", x=-0.005)
title_fix = fig.text(0.95, 1, " ", va="center", rotation="vertical") # Matplotlib is not user friendly

plt.savefig(figures_dir + f"MLE_per_min_dist_{n}_{''.join(model_groupings)}_untitled.pdf", bbox_extra_artists=(titlex, ttiley, title_fix), bbox_inches='tight')

# %%
# Plotting MFPT by Min Distance
cgt.plotting.latex_figure_setup()
if MODEL_NAME in ["SIE", "SIU"]:
    fig, axes = plt.subplots(3, 3)
    axes = axes.reshape(1, 9)[0]
    fig.set_size_inches(5, 4.5)
elif MODEL_NAME in ["IE", "IU"]:
    fig, axes = plt.subplots(2,3)
    axes = axes.reshape(1, 6)[0]
    fig.set_size_inches(5, 3)
fig.tight_layout()
fig.set_dpi(300)

results = simulation_results["models"][MODEL_NAME]["MFPTs_per_min_dist"]
for i, ax in enumerate(axes):
    vals = slice(i, (i + 2))
    values = list(results.values())[vals]
    all_vals = [val for sublist in values for val in sublist]
    labels = list(results.keys())[vals]
    boxes = ax.boxplot(
        values,
        widths=0.5,
        patch_artist=True,
        flierprops=dict(marker="o", markersize=1, linewidth=0.1),
    )
    for b, box in enumerate(boxes["boxes"]):
        box.set(facecolor=colors[i + b], linewidth=1)
    ax.set_xticklabels(labels)
    offset = (np.round(max(all_vals), 0) - np.round(min(all_vals), 0)) / 8
    ax.set_ylim(
        [np.round(min(all_vals), 0) - offset, np.round(max(all_vals), 0) + offset]
    )
    ax.set_yticks([np.round(min(all_vals), 0), np.round(max(all_vals), 0)])
    ax.tick_params(axis="y", labelsize=8, pad=0.5)
    ax.tick_params(axis="x", length=0)

fig.text(0.5, -0.03, "Minimum Distance", ha="center")
fig.text(-0.04, 0.5, "Mean First Passage Time (MFPT)", va="center", rotation="vertical")
fig.text(0.95, 1, " ", va="center", rotation="vertical")

#fig.suptitle(f"MFPT per Min Distance for $M_{{{MODEL_NAME}}}$ on {n} regions", y=1.05)
fig.savefig(
    figures_dir + f"MFPT_per_min_dist_{n}_{MODEL_NAME}_untitled.pdf", bbox_inches="tight"
)

# %%
# Comparing IE to IU

# Plot MFPTs for IE on one axis and MFPTs for IU on the other
MLE_IE = simulation_results["models"]["IE"]["mles"]
MLE_IU = simulation_results["models"]["IU"]["mles"]

MPFT = {}
for genome in MLE_IE.keys():
    a = MLE_IE[genome]
    b = MLE_IU[genome]
    if not np.isnan(a) and not np.isnan(b) and a != 0 and b != 0:
        MPFT[genome] = (a, b, a - b)

fig, axes = plt.subplots(1, 2)
fig.tight_layout()
fig.subplots_adjust(left=0.13, right=0.96, top=0.95, bottom=0.22, wspace=0.39)

# add some space on the right
fig.set_size_inches(5, 2.1)
fig.set_dpi(300)

# Plot histogram
values = [x[2] for x in MPFT.values()]
axes[0].hist(values, color=colors[4], edgecolor="black", linewidth=0.5)
axes[0].set_xlabel("Difference of MLEs")
axes[0].set_ylabel("Number of genomes", labelpad=7)
axes[0].set_xlim(-15, 15)

remove_duplicates = list(set(MPFT.values()))
axes[1].scatter(
    [x[0] for x in remove_duplicates],
    [x[1] for x in remove_duplicates],
    color=colors[4],
    edgecolor="black",
    linewidth=0.3,
    s=20,
)
# add a dotted line from corner to corner
x = np.linspace(0, 22, 100)
axes[1].plot(x, x, linestyle="dotted", color="black")
axes[1].set_xlabel("MLE under $M_{{IE}}$")
axes[1].set_ylabel("MLE under $M_{{IU}}$", labelpad=7)
axes[1].set_xlim(0, 22)
axes[1].set_ylim(0, 22)
axes[1].set_xticks(range(0, 23, 5))
axes[1].set_yticks(range(0, 23, 5))

fig.savefig(figures_dir + f"model_comparison_IE_IU_{n}_untitled.pdf")

# %%
