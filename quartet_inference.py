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
# Directory for results
results_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/results/complete_distributions/"
figures_dir = "/home/joshua/GitHub/genome-rearrangement-simulations/simulations_for_thesis/figures/new_figures/"
results_file_name = f"distribution_results_n{n}.pkl"

# %%
# Read results from the file
with open(results_dir + f"complete_distance_distribution_n{n}.pkl", "rb") as f:
    complete_distance_distribution = pickle.load(f)


#%%
def time_to_string(t):
    m, s = divmod(int(t), 60)
    return f"\tTook {m:02d} minutes and {s:02d} seconds"


def time_and_print(function):
    def wrapper(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        print(time_to_string(time() - t0))
        return result
    return wrapper

@time_and_print
def tree_setup():
    if simulation["internal_branch_length"] is None or simulation["leaf_branch_lengths"] is None:
        try:
            external = (round(uniform(*simulation['branch_lengths']),2) for _ in range(4))
            internal = round(uniform(*simulation['branch_lengths']),2)
        except TypeError:
            external = (simulation['branch_lengths'] for _ in range(4))
            internal = simulation['branch_lengths']
    else:
        internal = simulation["internal_branch_length"]
        external = (simulation["leaf_branch_lengths"] for _ in range(4))
    newick = f"((A:{next(external)},B:{next(external)}):{float(internal/2)},(C:{next(external)},D:{next(external)}):{float(internal/2)});"
    tree = cgt.simulations.newick_to_tree(newick)
    return tree

def compute_metrics(D_s, criterion):
    if criterion == 'smallest_pair':
        tuples = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        dists = {
            tup : D_s[*tup] for tup in tuples
        }
    elif criterion == 'four_point':
        dists = {
            (0,1) : D_s[0,1] + D_s[2,3],
            (0,2) : D_s[0,2] + D_s[1,3],
            (0,3) : D_s[0,3] + D_s[1,2]
        }
    return dists

def refresh_simulation(simulation):
    simulation['results'] = {method : [] for method in simulation['distances']}
    if simulation['leaf_branch_lengths'] is None and simulation['internal_branch_length'] is None:
        branch_length_string = str(simulation['branch_lengths']).replace(' ', '')
    else:
        branch_length_string = f"int{simulation['internal_branch_length']}_leaf{simulation['leaf_branch_lengths']}"
    name_parts = [
        simulation['tree'], 
        str(simulation['regions']), 
        branch_length_string,
        str(simulation['simulation_model']),
        str(simulation['estimation_model']),
    ]
    simulation['name'] = "_".join(name_parts)

def get_models(simulation):
    sim_model = models[simulation["simulation_model"]]
    est_model = models[simulation["estimation_model"]]
    return sim_model, est_model

def do_simulation(simulation, framework, sim_model, est_model, exact_bl_changes=False):
    for iteration in range(simulation["iterations"]):
        print(f"Simulation iteration {iteration}")
        t0 = time()

        print(f"\tGenerating tree...")
        tree = tree_setup()

        print(f"\tEvolving genomes on tree...")
        simulated_tree = cgt.simulations.evolve_on_tree(tree, framework, sim_model, exactly_bl_changes=exact_bl_changes)
        root = [n for n, d in simulated_tree.in_degree() if d == 0][0]
        cherry = sorted([list(simulated_tree.successors(x)) for x in simulated_tree.nodes if x != root])[-1]
        true_cherry = { simulated_tree.nodes[x]["label"] for x in cherry } # The paired genomes
        leaves = [n for n, d in simulated_tree.out_degree() if d == 0]
        genomes = [simulated_tree.nodes[leaf]["genome"] for leaf in leaves]
        labels = [simulated_tree.nodes[leaf]["label"] for leaf in leaves]
        print(f"\tGenomes at leaves: {labels}")

        for m, method in enumerate(simulation['distances']):
            print(f"({m}) Estimating tree using {str(method)}...")
            method_results = {}
            need_distances = distances.genomes_for_dist_matrix(
                framework, 
                genomes
            )
            pairs, need_distances = map(list, zip(*need_distances.items()))

            # Construct the distance matrix
            D = np.zeros((len(genomes), len(genomes)))
            for p, (i, j) in enumerate(pairs):
                D[i, j] = complete_distance_distribution[simulation["estimation_model"]][need_distances[p]][method]

            D_s = D + D.T # Make symmetric

            for criterion in simulation['criteria']:
                print(f"\tCriterion: {criterion}")
                method_results[criterion] = {}
                for replace_nan_with in simulation['replace_nan_with']:
                    method_results[criterion][replace_nan_with] = {}
                    print(f"\t\tReplace NaN with: {replace_nan_with}")
                    D_s_c = D_s.copy()
                    D_s_c[np.isnan(D_s_c)] = replace_nan_with if replace_nan_with != "NaN" else np.nan
                    dists = compute_metrics(D_s_c, criterion)
                    dists_without_nan = {k: v for k, v in dists.items() if not np.isnan(v)}
                    no_signal = False
                    if len(dists_without_nan) == 0:
                        tie = True
                        estimated_cherry = None
                        tied_pairs = [k for k,v in dists.items()]
                        no_signal = True
                    else:
                        es = min(dists_without_nan, key=dists_without_nan.get)
                        tied_pairs = [k for k,v in dists_without_nan.items() if v == dists_without_nan[es]]
                        tie = len(tied_pairs) > 1
                    # If it's a tie, select a random one from tied_pairs
                    was_tied = False
                    if tie:
                        was_tied = True
                        es = tied_pairs[np.random.randint(0, len(tied_pairs))]
                        tie = False
                    estimated_cherry = {labels[es[0]], labels[es[1]]}

                    method_results[criterion][replace_nan_with] = {
                        "simulated_tree": labels,
                        "simulated_cherry": true_cherry,
                        "estimated_cherry": estimated_cherry,
                        "estimated_distance_matrix": D,
                        "topology_correct": 'Tie' if tie else len(true_cherry & estimated_cherry) in (0, 2),
                        "was_tied": was_tied,
                        "no_signal" : no_signal
                    }
            simulation['results'][method].append(method_results)
            print(time_to_string(time() - t0))
            t0 = time()


# %% 
# Simulation on a quartet tree
criteria = ( 'four_point', "smallest_pair")
framework = fw
simulation = {
    "tree": "New_Quartet",
    "regions": n,
    "branch_lengths": None,
    "internal_branch_length": None,
    "leaf_branch_lengths": None,
    "criteria" : criteria,
    "simulation_model": "SIU",
    "estimation_model": "SIU",
    "replace_nan_with": ("NaN", 50, 100, 200),
    "distances": (
        "min",
        "MFPT",
        "MLE",
    ),
    "iterations": 1_000,
}

# %% Begin Simulation
for i in list(range(1, 12)):
    # Update simulation object
    simulation['internal_branch_length'] = 2
    simulation['leaf_branch_lengths'] = i
    refresh_simulation(simulation) # updates name and clears results
    sim_model, est_model = get_models(simulation)
    exact_bl_changes = True
    do_simulation(simulation, framework, sim_model, est_model, exact_bl_changes)
    # Save results
    file_name = results_dir + simulation['name'] + ('_e' if exact_bl_changes else '') + '.pickle'
    with open(file_name, "wb") as f:
        pickle.dump(simulation, f)


 # %%
chart_data = {}
criterion = 'four_point'
replace_nan_with = "NaN"

for i in list(range(1,12)):
    chart_data[i] = {}
    file_name = results_dir + f"New_Quartet_7_int2_leaf{i}_SIU_SIU_e.pickle"
    with open(file_name, "rb") as f:
        loaded_simulation = pickle.load(f)
    for method in loaded_simulation['results']:
        counts = Counter([
            x[criterion][replace_nan_with]['topology_correct'] for x in loaded_simulation['results'][method]
        ])
        print(counts)
        chart_data[i][method] = {
            'correct': counts[True] / len(loaded_simulation['results'][method]) * 100,
            'incorrect': counts[False] / len(loaded_simulation['results'][method]) * 100,
            'tie': counts['Tie'] / len(loaded_simulation['results'][method]) * 100,
        }
        print(chart_data[i][method])
        
cgt.plotting.latex_figure_setup()
fig, axes = plt.subplots()
fig.set_size_inches(5, 2.5)
fig.tight_layout()
fig.subplots_adjust(right=0.95, bottom=0.2, top=0.98, left=0.11)
fig.set_dpi(300)
methods = ["min", "MFPT", "MLE"]
for i, method in enumerate(methods):
    labels = [x for x in chart_data.keys()]
    values = [chart_data[i][method]['correct'] for i in labels]

    # plot clustered bar chart with offsets
    width = 0.23
    multiplier = i - len(loaded_simulation['results']) / 2.9
    offset = width * multiplier
    plt.bar(
        [x + offset for x in range(1, len(labels) + 1)],
        values,
        width=width,
        label = str(method).replace("DISTANCE.", "") if method != DISTANCE.min else "Min dist.",
        # use a pattern for colour
        color = plt.cm.Set2(i),
        edgecolor="black",
        linewidth=0.5,
    )
plt.xticks(range(1, len(labels) + 1), labels)
plt.yticks(range(0, 101, 20))
plt.xlabel("Branch length $\lambda$")
plt.ylabel("Proportion quartets correct")
# Remove border from legend
plt.legend(frameon=False)
plt.savefig(figures_dir + "quartet_results_NEWNEW.pdf")
plt.show()

# %%
