import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import sys

sys.setrecursionlimit(2000)


def plot_graph(G):
    # print(G.edges(data=True))
    # Create a list of node positions
    pos = nx.spring_layout(G)

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=750, node_color='skyblue', font_size=10, font_weight='bold')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=.5, font_size=8, font_color='red')

    # Show the plot
    plt.show()


def plot_solved_graph(G):
    ordi = copy.deepcopy(G)

    for e0, e1 in ordi.edges():
        if ordi[e0][e1].get("nb_cross") == 0:
            ordi.remove_edge(e0, e1)

    plot_graph(ordi)



def calc_graph_result(G):
    weights = 0

    for e0, e1 in G.edges():
        if G[e0][e1].get("nb_cross"):
            weights += G[e0][e1]["weight"]

    return weights


def check_crossed(G):
    nodes = list(nx.nodes(G))

    for e0 in G.nodes():
        if G.nodes[e0].get("nb_visit"):
            if e0 in nodes:
                nodes.remove(e0)
    return len(nodes) == 0


def deep_search(G, node0: int = 0, weights: int = 0):
    # condition d'arret : on verifie que tous les noeud aient ete visites
    if check_crossed(G):
        G.solution = weights
        print("RESULT", weights)
        return G

    answer = G
    # on recupere les voisins par ordre de proximite
    neighs = G.nodes[node0]['neighbors']

    len_neighs = len(neighs)

    for node1 in neighs:
        if len_neighs > 1:
            if G[node0][node1].get("nb_cross") > 1 or G.nodes[node1].get("nb_visit") > 1:
                continue

        ordi = copy.deepcopy(G)

        ordi[node0][node1]["nb_cross"] += 1
        n_weight = weights + ordi[node0][node1]["weight"]

        # Si le nouveau chemin est plus long que le meilleur
        if G.absolute_best <= n_weight:
            continue

        res = deep_search(ordi, node1, n_weight)

        if res.solution < G.absolute_best:
            G.absolute_best = res.solution
            answer = res

    return answer


def create_graph(size: int = 7, seed: int = None, fully_connected: bool = True):
    if seed is not None:
        random.seed(seed)
    # Create a complete graph with 15 nodes
    G = nx.complete_graph(size)

    i = -50

    if fully_connected:
        i = 1

    # Set the edge weights to random distances between 1 and 100
    for e0, e1 in G.edges():
        weight = random.randint(i, 100)
        if weight > 0:
            G[e0][e1]['weight'] = weight
            G[e0][e1]["nb_cross"] = 0

        else:
            G.remove_edge(e0, e1)

    for node in G.nodes():
        G.nodes[node]['neighbors'] = sorted(G.neighbors(node), key=lambda neighbor: G[node][neighbor]['weight'])
        G.nodes[node]["nb_visit"] = 0

    return G


def create_custom_graph():
    """
    G = nx.Graph()
    G.add_nodes_from(range(0, 5))

    G.add_edge(0, 3, weight=3)
    G.add_edge(1, 3, weight=3)
    G.add_edge(2, 3, weight=3)
    G.add_edge(4, 3, weight=3)
    G.add_edge(5, 3, weight=3)
    """
    G = nx.complete_graph(5)

    for e0, e1 in G.edges():
        G[e0][e1]["nb_cross"] = 0

        if e0 == 3 or e1 == 3:
            G[e0][e1]['weight'] = 1
        else:
            G[e0][e1]['weight'] = 50

    for node in G.nodes():
        G.nodes[node]['neighbors'] = sorted(G.neighbors(node), key=lambda neighbor: G[node][neighbor]['weight'])
        G.nodes[node]["nb_visit"] = 0

    return G



def solve_graph(G, best_solution: int = None, best_model=None):
    if best_model is not None and best_solution is not None:
        print(best_solution)
        plot_solved_graph(best_model)
    else:
        best_solution = float("inf")

    G.solution = best_solution
    G.absolute_best = best_solution
    ordi = deep_search(G)

    print(ordi.solution)
    plot_solved_graph(ordi)



# G = create_custom_graph()

G = create_graph(5, seed=1, fully_connected=True)

plot_graph(G)
solve_graph(G)
