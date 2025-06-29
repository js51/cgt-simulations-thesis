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
# Directory for results
results_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/results/complete_distributions/"
figures_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/figures/"

#%%
# Define frameworks
n = 7
fw = cgt.PositionParadigmFramework(n)

# %%
# Set up simulation models
sim_model_dicts = {
    #"IE":  {MODEL.all_inversions: 1},
    "IU":  {MODEL.all_inversions_larger_less_likely: 1},
    "SIE": {MODEL.one_region_inversions: 1 / 2, MODEL.two_region_inversions: 1 / 2},
    "SIU": {MODEL.one_region_inversions: 2 / 3, MODEL.two_region_inversions: 1 / 3},
    "T":   {MODEL.all_transpositions: 1},
    "AT":  {MODEL.two_region_adjacent_transpositions: 1},
}

model_names = list(sim_model_dicts.keys())

models = { name : cgt.Model.named_model_with_relative_probs(fw, sim_model_dicts[name]) for name in model_names }

# %%
# Compute distances for every genome by taking one per equivalence class
equiv_classes = fw.canonical_double_cosets(join_inverse_classes=True)

#%%
results = {
    model_name : {} for model_name in sim_model_dicts
}

distances_functions = { 
    "MLE" : cgt.distances.mles,
    "MFPT" : cgt.distances.fast_MFPT,
    "min" : cgt.distances.min_distance_using_irreps,
}

# For each model
for model_name, model in models.items():
    print(f"Processing model: {model_name}")
    # For each equivalence class
    for equiv_class in equiv_classes:
        print(f"Processing equivalence class of size {len(equiv_class)}")
        representative = next(iter(equiv_class))
        # For each distance function, compute the distance for the representative genome
        computed_distances = {}
        for dist_name, dist_func in distances_functions.items():
            print(f"\tComputing {dist_name} distance for representative {representative}")
            distance = dist_func(fw, model, [representative])[representative]
            computed_distances[dist_name] = distance
            print(f"\t\t{dist_name} distance: {distance}")
        # Set every genome to this distance
        for canonical_instance in equiv_class:
            results[model_name][canonical_instance] = computed_distances
    
    # Save the results for this model to a file (in case the computation takes a long time and fails)
    with open(results_dir + f"complete_distance_distribution_{model_name}_n{n}.pkl", "wb") as f:
        pickle.dump(results[model_name], f)


# %%
# Save complete results to a file
with open(results_dir + f"complete_distance_distribution_n{n}.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
# Read results from the file
with open(results_dir + f"complete_distance_distribution_n{n}.pkl", "rb") as f:
    complete_distance_distribution = pickle.load(f)

# %%
# Read results from the file (specific to the model)
model_name = "AT"  # Change this to the model you want to read
with open(results_dir + f"complete_distance_distribution_{model_name}_n{n}.pkl", "rb") as f:
    complete_distance_distribution_single_model = pickle.load(f)
