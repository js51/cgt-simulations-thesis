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

# %%
# Print versions of cgt and splitp
print(f'cgt version is {pkg_resources.require("cgt")[0].version}')
print(f'splitp version is {pkg_resources.require("splitp")[0].version}')


# %%
# Choose n
n = 4

# %%
# Directory for results
results_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/results/complete_distributions/"
figures_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/figures/"
results_file_name = f"distribution_results_n{n}.pkl"

# %%
# Read results from the file
with open(results_dir + f"complete_distance_distribution_n{n}.pkl", "rb") as f:
    complete_distance_distribution = pickle.load(f)


# %%
# Compute model diameters
diameters = {}
for model_name, model_result in complete_distance_distribution.items():
    diameter = 0
    for genome, distances in model_result.items():
        distance = distances['min']
        if distance > diameter:
            diameter = distance
    diameters[model_name] = diameter

print("Model diameters:")
for model_name, diameter in diameters.items():
    print(f"{model_name}: {diameter}")


# %% Compute proportion of genomes with an MLE (n=4 to 7)

mles_n = {}

for n in range(3, 8):
    with open(results_dir + f"complete_distance_distribution_n{n}.pkl", "rb") as f:
        complete_distance_distribution = pickle.load(f)
    mle = {}
    for model_name, model_result in complete_distance_distribution.items():
        mle_count = 0.0
        total_count = 0.0
        for genome, distances in model_result.items():
            if not np.isnan(distances['MLE']):
                mle_count += 1
            total_count += 1
        mle[model_name] = mle_count / total_count if total_count > 0 else 0
    mles_n[n] = mle

# Print the proportions of genomes with an MLE for each n and model
# as a latex table with n as columns and models as rows
#%%
for model_name in ["IE", "IU", "SIE", "SIU", "T", "AT"]:
    print(f"$M_{{{model_name}}}$", end=' ')
    for n_val in range(3, 8):
        print(" &", end=' ')
        print(f"{mles_n[n_val][model_name]:.1%}", end='')
    print(" \\\\ ")

# %%
