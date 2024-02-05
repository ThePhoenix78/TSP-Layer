import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from shapely.geometry import LineString

import random
import math
import copy
import sys

sys.setrecursionlimit(2000)


def check_crossed(G):
    nodes = list(nx.nodes(G))

    for e0 in G.nodes():
        if G.nodes[e0]["end_visit"]:
            if e0 in nodes:
                nodes.remove(e0)

    return len(nodes)


def is_on_trajectory(dot1, dot2, point):
    dxc = point[0] - dot1[0]
    dyc = point[1] - dot1[1]

    dxl = dot2[0] - dot1[0]
    dyl = dot2[1] - dot1[1]

    cross = (dxc * dyl) - (dyc * dxl)
    cross = round(cross, 1)

    if cross == 0:
        if (abs(dxl) >= abs(dyl)):
            if dxl > 0:
                return (dot1[0] <= point[0] and point[0] <= dot2[0])
            else:
                return (dot2[0] <= point[0] and point[0] <= dot1[0])

        else:
            if dyl > 0:
                return (dot1[1] <= point[1] and point[1] <= dot2[1])
            else:
                return (dot2[1] <= point[1] and point[1] <= dot1[1])

    return False


def intersect(lineA, lineB):
    line = LineString(lineA)
    other = LineString(lineB)
    return line.intersects(other)



def gen_coords(radius: int):
    angle = np.linspace(0, 2 * np.pi, 10000)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    return [[x[i], y[i]] for i in range(len(x))]


def calc_dist(node1, node2):
    return math.dist(node1, node2)


def calc_dist_line_dot(node1, node2, dot):
    return np.abs(np.cross(node2-node1, node1-dot)) / np.linalg.norm(node2-node1)


def get_dist_and_vector_dots(node1, node2, radius: int = 5, coords: list = None):
    if coords is None:
        coords = gen_coords(radius)

    for i in range(len(coords)):
        dot = coords[i]

        if is_on_trajectory(node1["position"], dot, node2["position"]):
            return calc_dist(node1["position"], node2["position"]), dot


def get_exterior_dots(G, dots):
    c = G.centers

    weight = 0
    starting_node = 0

    for i in dots:
        w = G.nodes[i]["c_weight"]

        if w > weight:
            starting_node = i
            weight = w

    return starting_node


def get_neighbors(G, node):
    liste = []
    neighs = sorted(G.neighbors(node), key=lambda n: G[node][n]['weight'])

    for n in neighs:
        if G[node][n]["nb_cross"]:
            liste.append(n)

    return liste


def is_enclaved(G, n):
    neighs = get_neighbors(G, n)
    s = len(neighs)

    for e in neighs:
        if G.nodes[e]["end_visit"] >= 1:
            s -= 1

    return s < 1


def check_enclaved(G):
    for node in G.nodes:
        if G.nodes[node]["end_visit"] == 0:
            if is_enclaved(G, node):
                return node

    return False


def open_node(G, node):
    neighs = list(G.neighbors(node))
    problem = []

    for n in neighs:
        lineA = [G.nodes[node]["position"], G.nodes[n]["position"]]

        for e0, e1 in G.edges():
            if G[node][n] == G[e0][e1] or G[e0][e1]["nb_cross"] == 0 or node in [e0, e1] or n in [e0, e1]:
                continue

            lineB = [G.nodes[e0]["position"], G.nodes[e1]["position"]]

            if intersect(lineA, lineB):
                problem.append(n)
                break

    for n in neighs:
        if n not in problem:
            G[node][n]["nb_cross"] += 1

    return G


def plot_graph(G):
    # Draw the graph with the positions of the nodes
    G = copy.deepcopy(G)

    for e0, e1 in G.edges():
        if G[e0][e1].get("nb_cross") == 0:
            G.remove_edge(e0, e1)

    nx.draw(G, [G.nodes[i]["position"] for i in G.nodes], with_labels=True, node_size=200, node_color='skyblue')

    # Show the plot
    plt.show()


def plot_solved_graph(G):
    # Draw the graph with the positions of the nodes
    G = copy.deepcopy(G)

    for e0, e1 in G.edges():
        if G[e0][e1]["end_cross"] == 0:
            G.remove_edge(e0, e1)

    nx.draw(G, [G.nodes[i]["position"] for i in G.nodes], with_labels=True, node_size=200, node_color='skyblue')

    # Show the plot
    plt.show()


def plot_graph_step(G):
    G = copy.deepcopy(G)
    pos = [G.nodes[i]["position"] for i in G.nodes]

    plt.clf()

    tour = []

    for n in G.nodes:
        if G.nodes[n]["nb_visit"]:
            tour.append(n)

    for e0, e1 in G.edges():
        if G[e0][e1]["nb_cross"] == 0:
            G.remove_edge(e0, e1)

    path_edges = list(zip(tour, tour[1:]))

    nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue')
    # nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)
    plt.pause(.1)


def plot_final_graph_step(G):
    G = copy.deepcopy(G)
    pos = [G.nodes[i]["position"] for i in G.nodes]

    plt.clf()

    for e0, e1 in G.edges():
        if G[e0][e1]["nb_cross"] == 0:
            G.remove_edge(e0, e1)

    path_edges = []

    for e0, e1 in G.edges():
        if G[e0][e1]["end_cross"] >= 1:
            path_edges.append([e0, e1])


    nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', width=.5)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2.5)
    plt.pause(.1)



def create_graph(size: int = 10, seed: int = None, nb_center: int = 1, random_state: int = None):
    if seed is not None:
        random.seed(seed)

    # Create an empty graph with no nodes and no edges
    # G = nx.empty_graph(size)
    G = nx.complete_graph(size)

    # Generate a cloud point with size points
    x, y, c = make_blobs(n_samples=size, centers=nb_center, random_state=random_state, return_centers=True)

    # save the center of the cluster
    G.centers = c

    weight = 0
    starting_node = 0

    # Add nodes to the graph
    for i in range(size):
        G.add_node(i)
        G.nodes[i]["position"] = x[i]
        G.nodes[i]["center"] = y[i]
        G.nodes[i]["nb_visit"] = 0
        G.nodes[i]["end_visit"] = 0
        G.nodes[i]["neighbors"] = []
        G.nodes[i]["no"] = i

        w = calc_dist(c[0], x[i])

        G.nodes[i]["c_weight"] = w
        G.nodes[i]["weight"] = None

        if w > weight:
            starting_node = i
            weight = w

    for e0, e1 in G.edges():
        G[e0][e1]['weight'] = calc_dist(G.nodes[e0]["position"], G.nodes[e1]["position"])
        G[e0][e1]["nb_cross"] = 0
        G[e0][e1]["end_cross"] = 0
        G[e0][e1]["vector"] = [0, 0]
        G[e0][e1]["layer"] = None

    G.solution = float("inf")
    G.absolute_best = float("inf")

    G.starting_node = starting_node
    G.radius = round(weight, 0) * 5
    G.layers = {}

    return G


def create_layers(G, starting_node: int = 0, depth: int = 0, coords: list = None):
    if coords is None:
        coords = gen_coords(radius=G.radius)

    j = starting_node

    my_nodes = [i for i in range(len(G.nodes)) if not G.nodes[i]["nb_visit"]]

    over = False

    initial_point = G.nodes[j]["no"]
    center = G.nodes[j]
    G.nodes[j]["nb_visit"] += 1

    layer = [starting_node]

    while not over:
        for i in range(len(coords)):
            if over:
                break

            dot = coords[i]

            for n in my_nodes:
                trajectory = is_on_trajectory(center["position"], dot, G.nodes[n]["position"])
                if trajectory and n != center["no"]:
                    if G.nodes[n]["no"] == initial_point:
                        weight = calc_dist(G.nodes[j]["position"], G.nodes[n]["position"])

                        G[j][n]["vector"] = dot
                        G[j][n]["nb_cross"] += 1
                        G[j][n]["layer"] = depth

                        over = True
                        break

                    elif not G.nodes[n]["nb_visit"]:
                        weight = calc_dist(center["position"], G.nodes[n]["position"])

                        G.nodes[n]["nb_visit"] += 1
                        G.nodes[n]["weight"] = weight

                        center = G.nodes[n]

                        G[j][n]["vector"] = dot
                        G[j][n]["nb_cross"] += 1
                        G[j][n]["layer"] = depth

                        layer.append(n)
                        j = n

    not_crossed = [i for i in range(len(G.nodes)) if not G.nodes[i]["nb_visit"]]

    G.layers[depth] = layer

    if len(not_crossed) > 2:
        return create_layers(G, get_exterior_dots(G, not_crossed), depth + 1, coords)
    else:
        for n in not_crossed:
            smallest = []
            min_dist = float("inf")

            for e0, e1 in G.edges():
                if (e0 not in layer) or (e1 not in layer) or G[e0][e1]["nb_cross"] == 0:
                    continue

                dist = calc_dist_line_dot(G.nodes[e0]["position"], G.nodes[e1]["position"], G.nodes[n]["position"])

                if dist < min_dist:
                    min_dist = dist
                    smallest = [e0, e1]

            G[n][smallest[0]]["nb_cross"] += 1
            G[n][smallest[0]]["layer"] = depth

            G[n][smallest[1]]["nb_cross"] += 1
            G[n][smallest[1]]["layer"] = depth

            G[smallest[0]][smallest[1]]["nb_cross"] = 0
            G[smallest[0]][smallest[1]]["layer"] = None

        G.layers[depth].extend(not_crossed)

    return G


def link_layers(G):
    smallest = []
    j = 0

    for i in range(len(G.layers.keys())-1):
        layer0, layer1 = G.layers[i], G.layers[i+1]

        for node in layer1:
            smallest.append([node, []])
            min_dist = float("inf")

            for e0, e1 in G.edges():
                if (e0 not in layer0) or (e1 not in layer0) or G[e0][e1]["nb_cross"] == 0:
                    continue

                dist = calc_dist_line_dot(G.nodes[e0]["position"], G.nodes[e1]["position"], G.nodes[node]["position"])

                if dist < min_dist:
                    min_dist = dist
                    smallest[j][1] = [e0, e1]

            j += 1

    for e in smallest:
        G[e[0]][e[1][0]]["nb_cross"] += 1
        G[e[0]][e[1][1]]["nb_cross"] += 1

        # G[e[1][0]][e[1][1]]["nb_cross"] = 0

        if G[e[0]][e[1][0]]["weight"] < G[e[1][0]][e[1][1]]["weight"] and G[e[0]][e[1][1]]["weight"] < G[e[1][0]][e[1][1]]["weight"]:
            G[e[1][0]][e[1][1]]["nb_cross"] = 0

    return G


def uncross_lines(G):
    for e0, e1 in G.edges():
        if G[e0][e1]["nb_cross"] == 0:
            continue

        lineA = [G.nodes[e0]["position"], G.nodes[e1]["position"]]

        for n0, n1 in G.edges():
            if G[n0][n1]["nb_cross"] == 0 or G[e0][e1] == G[n0][n1]:
                continue

            elif n0 in [e0, e1] or n1 in [e0, e1]:
                continue

            lineB = [G.nodes[n0]["position"], G.nodes[n1]["position"]]

            if intersect(lineA, lineB):
                print(e0, e1, "|", n0, n1, "->", e0, n1, "|", e1, n0, "/\\", e0, n0, "|", e1, n1)

                G[e0][e1]["nb_cross"] = 0
                G[n0][n1]["nb_cross"] = 0

                G[e0][n1]["nb_cross"] += 1
                G[e1][n0]["nb_cross"] += 1

                G[e0][n0]["nb_cross"] += 1
                G[e1][n1]["nb_cross"] += 1

    return G


def tunnel_node_list(G, nodes: list):
    neighs = []

    for node in nodes:
        neighs.append(get_neighbors(G, node))

    neighs0 = get_neighbors(G, node0)
    neighs1 = get_neighbors(G, node1)

    for n in neighs0:
        if n == node1:
            continue

        if G[node0][n]["weight"] > G[node1][n]["weight"]:
            G[node0][n]["nb_cross"] = 0
            G[node1][n]["nb_cross"] += 1
        else:
            G[node0][n]["nb_cross"] += 1
            G[node1][n]["nb_cross"] = 0

    for n in neighs1:
        if n == node0:
            continue

        if G[node0][n]["weight"] > G[node1][n]["weight"]:
            G[node0][n]["nb_cross"] = 0
            G[node1][n]["nb_cross"] += 1
        else:
            G[node0][n]["nb_cross"] += 1
            G[node1][n]["nb_cross"] = 0

    return G


def tunnel_nodes(G, node0, node1):
    neighs0 = get_neighbors(G, node0)
    neighs1 = get_neighbors(G, node1)

    for n in neighs0:
        if n == node1:
            continue

        if G[node0][n]["weight"] > G[node1][n]["weight"]:
            G[node0][n]["nb_cross"] = 0
            G[node1][n]["nb_cross"] += 1
        else:
            G[node0][n]["nb_cross"] += 1
            G[node1][n]["nb_cross"] = 0

    for n in neighs1:
        if n == node0:
            continue

        if G[node0][n]["weight"] > G[node1][n]["weight"]:
            G[node0][n]["nb_cross"] = 0
            G[node1][n]["nb_cross"] += 1
        else:
            G[node0][n]["nb_cross"] += 1
            G[node1][n]["nb_cross"] = 0

    return G


def tunnel_one(G, node0, thresold: float = .5):
    neighs0 = get_neighbors(G, node0)

    for node1 in neighs0:
        if G[node0][node1]["weight"] > thresold:
            continue

        neighs1 = get_neighbors(G, node1)

        for n in neighs0:
            if n == node1:
                continue

            if G[node0][n]["weight"] > G[node1][n]["weight"]:
                G[node0][n]["nb_cross"] = 0
                G[node1][n]["nb_cross"] += 1
            else:
                G[node0][n]["nb_cross"] += 1
                G[node1][n]["nb_cross"] = 0

        for n in neighs1:
            if n == node0:
                continue

            if G[node0][n]["weight"] > G[node1][n]["weight"]:
                G[node0][n]["nb_cross"] = 0
                G[node1][n]["nb_cross"] += 1
            else:
                G[node0][n]["nb_cross"] += 1
                G[node1][n]["nb_cross"] = 0

    return G


def tunnel_all(G, thresold: float = 0.5):
    for e0, e1 in G.edges():
        if not G[e0][e1]["nb_cross"]:
            continue

        if G[e0][e1]["weight"] < thresold:
            # print(e0, e1)

            G[e0][e1]["nb_cross"] += 1

            e0_neighs = get_neighbors(G, e0)
            e1_neighs = get_neighbors(G, e1)

            for n in e0_neighs:
                if n == e1:
                    continue

                if G[e0][n]["weight"] > G[e1][n]["weight"]:
                    G[e0][n]["nb_cross"] = 0
                    G[e1][n]["nb_cross"] += 1
                else:
                    G[e0][n]["nb_cross"] += 1
                    G[e1][n]["nb_cross"] = 0

            for n in e1_neighs:
                if n == e0:
                    continue

                if G[e0][n]["weight"] > G[e1][n]["weight"]:
                    G[e0][n]["nb_cross"] = 0
                    G[e1][n]["nb_cross"] += 1
                else:
                    G[e0][n]["nb_cross"] += 1
                    G[e1][n]["nb_cross"] = 0

    return G


def protect_alone(G):
    alone = []

    for n in G.nodes:
        G.nodes[n]["neighbors"] = get_neighbors(G, n)

        if len(G.nodes[n]["neighbors"]) <= 2:
            alone.append(n)

    for n in G.nodes:
        if n in alone:
            continue

        neighbors_alone = []
        neighbors_connected = []

        for m in G.nodes[n]["neighbors"]:
            if m in alone:
                neighbors_alone.append(m)
            else:
                neighbors_connected.append(m)

        # print(n, G.nodes[n]["neighbors"], neighbors_alone, neighbors_connected)
        if len(neighbors_alone) >= 2:
            for m in neighbors_connected:
                G[n][m]["nb_cross"] = 0
                for k in neighbors_connected:
                    if m != k:
                        G[m][k]["nb_cross"] += 1

    return G


def deep_search(G, node0: int = 0, weights: int = 0):
    # condition d'arret : on verifie que tous les noeud aient ete visites
    if check_crossed(G) <= 1:
        if 0 not in get_neighbors(G, node0):
            G.solution = float("inf")
            return G

        G[node0][0]["end_cross"] += 1
        weights += G[node0][0]["weight"]

        G.solution = weights
        print("RESULT", weights)
        plot_final_graph_step(G)
        return G

    # answer = G
    neighs = get_neighbors(G, node0)

    # on marque le noeud comme visite
    G.nodes[node0]["end_visit"] += 1

    answer = G

    for node1 in neighs:
        if G.nodes[node1]["end_visit"]:
            continue

        ordi = copy.deepcopy(G)

        ordi[node0][node1]["end_cross"] += 1
        n_weight = weights + ordi[node0][node1]["weight"]

        # Si le nouveau chemin est plus long que le meilleur
        if G.absolute_best <= n_weight or ((check_crossed(G) <= 1) and G.absolute_best <= (n_weight + ordi[node1][0]["weight"])) or check_enclaved(ordi) or is_enclaved(G, 0):
            continue

        plot_final_graph_step(ordi)
        res = deep_search(ordi, node1, n_weight)

        if res.solution < G.absolute_best:
            G.absolute_best = res.solution
            answer = res

    return answer


def solve_graph(G):
    plot_graph(G)

    G = create_layers(G, G.starting_node)
    print("STEP 1")
    plot_graph(G)

    G = link_layers(G)
    print("STEP 2")
    plot_graph(G)

    G = uncross_lines(G)
    G = uncross_lines(G)
    print("uncross_lines")
    plot_graph(G)

    G = protect_alone(G)
    G = uncross_lines(G)
    print("STEP 5")
    plot_graph(G)

    """
    G = tunnel_all(G, .5)
    G = tunnel_all(G, .5)
    G = protect_alone(G)
    G = uncross_lines(G)
    print("tunnel_all")
    plot_graph(G)
    """

    # G = open_node(G, 0)
    # plot_graph(G)

    return deep_search(G)


G = create_graph(50, seed=None, random_state=1)
print("ready")


G = solve_graph(G)
input("OVER")
plot_solved_graph(G)
