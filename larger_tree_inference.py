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
from networkx import draw
import random

# %%
# Print versions of cgt and splitp
print(f'cgt version is {pkg_resources.require("cgt")[0].version}')
print(f'splitp version is {pkg_resources.require("splitp")[0].version}')


# %%
# Directory for results
results_dir = "/home/joshua/GitHub/cgt-simulations-thesis/data/"
figures_dir = "/home/joshua/GitHub/cgt-simulations-thesis/figures/"

#%%
# Define frameworks
n = 7
fw = cgt.PositionParadigmFramework(n)

# %%
# Set up simulation results dictionary
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
# Read results from the file
with open(results_dir + f"complete_distance_distribution_n{n}.pkl", "rb") as f:
    complete_distance_distribution = pickle.load(f)


# %% Define tree
newick_string = "(((Z,A),B),((X,Y),D));"
tree = cgt.simulations.newick_to_tree(newick_string)
sp.Phylogeny(newick_string).draw()


#%%
chosen_model = "SIE"

# %%
new_tree = cgt.simulations.evolve_on_tree(tree, fw, models[chosen_model], exactly_n_changes=1) # dont change


# %% Get the genomes at the leaves of the tree
leaves = [n for n, d in new_tree.out_degree() if d == 0]
genomes = [new_tree.nodes[leaf]["genome"] for leaf in leaves]
labels = [new_tree.nodes[leaf]["label"] for leaf in leaves]

def get_splits(tree):
    G = tree.copy()
    edges = list(G.edges)
    splits = []
    for edge in edges:
        G.remove_edge(*edge)
        split = [set({x for x in c if G.out_degree()[x] == 0}) for c in nx.weakly_connected_components(G)]
        if min(len(x) for x in split) > 1:
            splits.append(sorted(split))
        G.add_edge(*edge)
    splits = {
        tuple(sorted((
                tuple(sorted(leaves.index(p) if p in leaves else p for p in part)) for part in split
            ), key=sorted
        )) 
        for split in splits
    }
    return splits

correct_spits = sorted(get_splits(new_tree))



#%% Results
results = []

for i in range(10_000):
    print(f"Iteration {i}")
    # Generate new tree
    print(f"> Generating tree...")
    new_tree = cgt.simulations.evolve_on_tree(tree, fw, models[chosen_model], exactly_n_changes=1)
    #cgt.simulations.draw_tree(new_tree)

    # Get the genomes at the leaves of the tree
    leaves = [n for n, d in new_tree.out_degree() if d == 0]
    genomes = [new_tree.nodes[leaf]["genome"] for leaf in leaves]
    labels = [new_tree.nodes[leaf]["label"] for leaf in leaves]

    need_distances = distances.genomes_for_dist_matrix(
        fw, 
        genomes
    )
    pairs, need_distances = map(list, zip(*need_distances.items()))


    D_matrices = {}
    for method in ["min", "MLE", "MFPT"]:
        # Construct the distance matrix
        D = np.zeros((len(genomes), len(genomes)))
        for p, (i, j) in enumerate(pairs):
            D[i, j] = complete_distance_distribution[chosen_model][need_distances[p]][method]
        D_matrices[method] = D

    D_min = D_matrices["min"]
    D_MLE = D_matrices["MLE"]
    D_MFPT = D_matrices["MFPT"]
        
    # make symmetric
    D_min = (D_min + D_min.T)
    D_MLE = (D_MLE + D_MLE.T)
    D_MFPT = (D_MFPT + D_MFPT.T)

    D_MLE[np.isnan(D_MLE)] = 99
    D_MLE[D_MLE == np.inf] = 99 # replace inf with 99   

    # Compute NJ trees
    print(f"> Computing NJ trees...")
    min_tree = phylogenetics.neighbour_joining(D_min, labels=labels)
    MLE_tree = phylogenetics.neighbour_joining(D_MLE, labels=labels)
    MFPT_tree = phylogenetics.neighbour_joining(D_MFPT, labels=labels)

    # Compute splits
    print(f"> Computing splits...")
    min_splits = set(get_splits(min_tree))
    MLE_splits = set(get_splits(MLE_tree))
    MFPT_splits = set(get_splits(MFPT_tree))
    correct_spits = set(correct_spits)

    # Compare splits
    print(f"> Comparing splits...")
    min_score = len(min_splits & correct_spits)
    MLE_score = len(MLE_splits & correct_spits)
    MFPT_score = len(MFPT_splits & correct_spits)

    # Print scores
    print(f"Scores: min={min_score}, MLE={MLE_score}, MFPT={MFPT_score}")
    #print("D_MLE=\n", D_MLE.round(1))
    #print("D_MFPT=\n", D_MFPT.round(1))
    #print("D_min=\n", D_min.round(1))

    # Add to results
    results.append({
        "min": min_score,
        "MLE": MLE_score,
        "MFPT": MFPT_score
    })


# %%
# Count the winners (highest score)
winners = Counter()
for result in results:
    max_score = max(result.values())
    for method, score in result.items():
        if score == max_score:
            winners[method] += 1.0
print("Winners:")
for method, count in winners.items():
    print(f"{method}: {count} times ({count / len(results) * 100:.1f}%)")

# %%
# Average score per method
average_scores = {method: 0 for method in results[0].keys()}
for result in results:
    for method, score in result.items():
        average_scores[method] += score
average_scores = {method: ((0.0) + score) / len(results) for method, score in average_scores.items()}
print(average_scores)

# %%
# How many 3's are there in the results?
three_counts = {method: 0 for method in results[0].keys()}
for result in results:
    for method, score in result.items():
        if score >= 3:
            three_counts[method] += 1
print("Three counts:")
for method, count in three_counts.items():
    print(f"{method}: {count} times ({1.0 * count / len(results) * 100:.1f}%)")
# %%
rolling_average = {method: [] for method in results[0].keys()}
for i in range(len(results)):
    for method in results[0].keys():
        if i == 0:
            rolling_average[method].append(results[i][method])
        else:
            rolling_average[method].append(
                np.mean([result[method] for result in results[0:i + 1]])
            )

#%% Now do a rolling average of proportion of times the score was 3
rolling_average_proportion = {method: [] for method in results[0].keys()}
cumulative_success = {method: 0 for method in results[0].keys()}
for r, result in enumerate(results): 
    for method in results[0].keys():
        if result[method] >= 3:
            cumulative_success[method] += 1
        proportion = cumulative_success[method] / (r + 1)
        rolling_average_proportion[method].append(proportion)


# %%
# Plot the rolling average
plt.figure(figsize=(10, 6))
for method, scores in rolling_average.items():
    plt.plot(scores, label=method)
plt.xlabel("Iteration")
plt.ylabel("Rolling Average Score")
plt.title("Rolling Average of Scores")
plt.legend()
plt.grid()
plt.show()
# %%

# Plot the rolling average proportion of times the score was 3
plt.figure(figsize=(10, 6))
for method, scores in rolling_average_proportion.items():
    plt.plot(scores, label=method)
plt.xlabel("Iteration")
plt.ylabel("Rolling Average Proportion of Score >= 3")
plt.title("Rolling Average Proportion of Times Score >= 3")
plt.legend()
plt.grid()
plt.show()


# %%
