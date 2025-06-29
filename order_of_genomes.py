#%% Imports 
import cgt
from cgt import *
import cgt.simulations
import cgt.parsers
import splitp as sp
import numpy as np
from splitp import phylogenetics
from cgt import distances
import pkg_resources
import networkx as nx
from time import time
import pickle
from random import uniform
import matplotlib.pyplot as plt
from collections import Counter
import random
import matplotlib.ticker as mtick

# %%
# Print versions of cgt and splitp
print(f'cgt version is {pkg_resources.require("cgt")[0].version}')
print(f'splitp version is {pkg_resources.require("splitp")[0].version}')


# %%
# Directory for results
results_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/results/complete_distributions/"
figures_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/figures/complete_distributions/"

#%%
# Define frameworks
n = 7
fw = cgt.PositionParadigmFramework(n)

# %%
# Set up simulation models
sim_model_dicts = {
    "IE":  {MODEL.all_inversions: 1},
    "IU":  {MODEL.all_inversions_larger_less_likely: 1},
    "SIE": {MODEL.one_region_inversions: 1 / 2, MODEL.two_region_inversions: 1 / 2},
    "SIU": {MODEL.one_region_inversions: 2 / 3, MODEL.two_region_inversions: 1 / 3},
    "T":   {MODEL.all_transpositions: 1},
    "AT":  {MODEL.two_region_adjacent_transpositions: 1},
}

model_names = list(sim_model_dicts.keys())

models = { name : cgt.Model.named_model_with_relative_probs(fw, sim_model_dicts[name]) for name in model_names }


# %%
# Set up plotting
PRINT_TITLES = False

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


# %% 
# Load results from the file
with open(results_dir + f"complete_distance_distribution_n{n}.pkl", "rb") as f:
    complete_distance_distribution = pickle.load(f)

#%%
methods = ("MLE", "min", "MFPT")
model_name = "SIU"


# %%
identity = fw.identity_instance()
def compute_chain(n):
    genomes = []
    genomes.append(identity)
    all_distances = []
    for i in range(n + 1):
        genome = genomes[-1]
        new_genome = genome * cgt.simulations.draw_genome(fw.symmetry_element()) * cgt.simulations.draw_rearrangement(model)
        new_genome_instance = fw.canonical_instance(new_genome)
        genomes.append(new_genome)
        distances = {
            method : complete_distance_distribution[model_name][new_genome_instance][method]
            for method in methods
        }
        all_distances.append(distances)
    return all_distances


# %%
iterations = 50

#%%
# Compute many chains
runs = 100
model = models[model_name]  # Use the SIU model for the chains
all_chains = []
for i in range(runs):
    print(f"Chain {i+1} of {runs}\t", end="\r")
    all_chains.append(compute_chain(iterations))

#%%
# Compute averages and standard deviations
keys = list(all_chains[0][0].keys())
averages = { key : [] for key in keys }
standard_deviations = { key : [] for key in keys }
average_num_nan = []

for i in range(iterations):
    counts = { key : [] for key in keys }
    nans = []
    for run in all_chains: # out of 500
        for key in keys:
            val = run[i][key]
            if np.isnan(run[i][key]):
                if key == 'MLE': nans.append(0)
            else:
                counts[key].append(val)
                if key == 'MLE': nans.append(1)
    for key in keys:
        averages[key].append(np.mean(counts[key]))
        standard_deviations[key].append(np.std(counts[key]))
    average_num_nan.append(100 * np.mean(nans))

#%%
# Plot mean with error bars for each key, on different subplots stacked vertically, shared x axis
fig, axs = plt.subplots(len(keys) + 1, figsize=(5.5, 5.5), constrained_layout=True)
xlabs = range(1, iterations + 1)
for i, key in enumerate(('min', 'MLE', 'MFPT')):
    axs[i].errorbar(xlabs, averages[key], yerr=standard_deviations[key], fmt='.', markersize=4, capsize=1.5, linewidth=0.5)
    axs[i].tick_params(axis='x', bottom=True, labelbottom=False)
    if key == "min":
        title = "Minimum distance" 
    elif key == 'MLE':
        title = "Maximum Likelihood Estimate (MLE)"
    elif key == 'MFPT':
        title = "Mean first passage time (MFPT)"
    axs[i].set_title(title, fontsize=11, loc='right')
    axs[i].set_ylim(0, max(averages[key]) + max(standard_deviations[key]) + 1)
    axs[i].locator_params(axis='y', nbins=4) 
    axs[i].set_xlim(0, None)
    if key == 'MFPT':
        # Set ylimits to be min and max of averages
        axs[i].set_ylim(min(averages[key]) - 1000, max(averages[key]) + 1000)
axs[-1].scatter(xlabs, average_num_nan, c='black', s=4)
axs[-1].set_title('Proportion with MLE', fontsize=11, loc='right')
axs[-1].set_xlabel("Rearrangements applied")
axs[-1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
axs[-1].set_ylim(0, 100 + 10)
axs[-1].set_xlim(0, None)
axs[-1].locator_params(axis='x', nbins=16) 
axs[-1].locator_params(axis='y', nbins=3) 
axs[-1].yaxis.set_major_formatter(mtick.PercentFormatter())
# Print solid red line at 0.5
#axs[-1].axhline(y=50, color='red', linestyle='-', linewidth=0.5)
title_fix = fig.text(0.985, 0.5, " ", va="center", rotation="vertical") # Matplotlib is not user friendly
plt.savefig(figures_dir + f"genome_iteration_{model_name}.pdf")




# %%
# Plot output 
runs = len(all_chains)
successes = { 
    method : [runs] + [0 for _ in range(runs - 1)] for method in all_chains[0][0].keys()
}
nans = [0 for _ in range(runs)]

for run, all_distances in enumerate(all_chains):
    print(f"Run {run+1} of {runs}\t", end="\r")
    for i, d in enumerate(all_distances):
        for key in successes:
            if (d[key] >= all_distances[i-1][key]) or (np.isnan(d[key])):
                successes[key][i] += 1
            if np.isnan(d[key]) and key == 'MLE':
                nans[i] += 1
    

cgt.plotting.latex_figure_setup()

fig, ax = plt.subplots(figsize=(4.5, 2.8))

for method, data in successes.items():
    ax.plot(
        [ 100 * float(d) / runs for d in data ], 
        label=method.replace("min", "Min. distance")
    )
# Add a line for the proportion of MLEs
ax.plot(
    [ 100 * float(d) / runs for d in nans ], 
    label="\% no MLE", 
    linestyle='-', 
    color='red',
    markersize=0,
    linewidth=0.75
)

# Remove legend border
fig.legend(loc='upper right', frameon=False, fontsize=11, bbox_to_anchor=(1.28, 0.95))
ax.set_xlabel("Rearrangements applied")
ax.set_ylabel("\% pairs ordered correctly")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 100)
ax.set_xlim(0, iterations)
title_fix = fig.text(0.98, 0.5, " ", va="center", rotation="vertical") # Matplotlib is not user friendly
plt.savefig(figures_dir + "order_of_genomes.pdf", bbox_inches='tight')

# %%
