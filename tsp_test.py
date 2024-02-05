import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random
import math
import copy


def check_crossed(G):
    nodes = list(nx.nodes(G))

    for e0 in G.nodes():
        if G.nodes[e0].get("link") is not None:
            if e0 in nodes:
                nodes.remove(e0)

    return len(nodes) == 0


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


def get_dist_and_vector_dots(node1, node2, radius: int, coords):
    for i in range(len(coords)):
        dot = coords[i]

        if is_on_trajectory(node1["position"], dot, node2["position"]):
            return math.dist(node1["position"], node2["position"]), dot


def plot_graph(G):
    # Draw the graph with the positions of the nodes
    nx.draw(G, [G.nodes[i]["position"] for i in G.nodes], with_labels=True, node_size=200, node_color='skyblue')

    # Show the plot
    plt.show()


def create_graph(size: int = 10, seed: int = None, nb_center: int = 1):
    if seed is not None:
        random.seed(seed)

    # Create an empty graph with no nodes and no edges
    G = nx.empty_graph(size)
    # G = nx.complete_graph(size)

    # Generate a cloud point with 100 points
    x, y, c = make_blobs(n_samples=size, centers=nb_center, random_state=seed, return_centers=True)

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
        G.nodes[i]["no"] = i
        G.nodes[i]["link"] = None

        w = math.dist(c[0], x[i])

        G.nodes[i]["c_weight"] = w
        G.nodes[i]["weight"] = None

        if w > weight:
            starting_node = i
            weight = w

    G.starting_node = starting_node
    G.radius = round(weight, 0) * 3

    return G


def replace_dot(G, done, radius: int, coords):
    len_nodes = len(G.nodes)

    done = sorted(done, key=lambda i: G[i][G.nodes[i]["link"]]["weight"])
    print(done)

    for e0 in done:
        node = G.nodes[e0]

        if node["link"] is None:
            continue

        good_vector = False
        over = False

        for i in range(len(coords)):
            if over:
                break

            dot = coords[i]

            if dot != G[node["no"]][node["link"]]["vector"] and not good_vector:
                # coords.append(dot)
                continue

            good_vector = True


            for n in range(len_nodes):
                if n != node["no"] and is_on_trajectory(node["position"], dot, G.nodes[n]["position"]) and not G.nodes[n]["nb_visit"]:

                    weight = math.dist(node["position"], G.nodes[n]["position"])

                    if weight > G[node["no"]][node["link"]]["weight"]:
                        continue

                    G.nodes[n]["nb_visit"] += 1
                    n_weight, n_vector = get_dist_and_vector_dots(G.nodes[n], G.nodes[node["link"]], 20, coords)

                    G.add_edge(node["no"], n, weight=weight, vector=dot)
                    G.add_edge(n, node["link"], weight=n_weight, vector=n_vector)

                    G.remove_edge(node["no"], node["link"])

                    G.nodes[n]["link"] = node["link"]
                    node["link"] = n

                    done.append(n)
                    over = True
                    break


    return done


def solve_salesman(G, radius: int = 5, starting_node: int = 0):
    angle = np.linspace(0, 2 * np.pi, 3600)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    coords = [[x[i], y[i]] for i in range(len(x))]

    j = starting_node

    len_nodes = len(G.nodes)

    over = False
    done = [j]

    initial_point = G.nodes[j]["no"]
    center = G.nodes[j]
    G.nodes[j]["nb_visit"] += 1

    while not check_crossed(G) and not over:
        for i in range(len(x)):
            if over:
                break

            dot = coords[i]

            for n in range(len_nodes):
                trajectory = is_on_trajectory(center["position"], dot, G.nodes[n]["position"])

                if n != center["no"] and trajectory and G.nodes[n]["no"] == initial_point:
                    weight = math.dist(G.nodes[j]["position"], G.nodes[n]["position"])

                    center["link"] = n

                    G.add_edge(j, n, weight=weight, vector=dot)
                    over = True
                    break

                elif n != center["no"] and trajectory and not G.nodes[n]["nb_visit"]:
                    weight = math.dist(center["position"], G.nodes[n]["position"])

                    center["link"] = n

                    G.nodes[n]["nb_visit"] += 1
                    G.nodes[n]["weight"] = weight

                    center = G.nodes[n]

                    G.add_edge(j, n, weight=weight, vector=dot)

                    done.append(n)

                    j = n

    # print(G.nodes(data=True))
    print(done)
    plot_graph(G)

    if len(done) != len_nodes:
        done = replace_dot(G, done, radius, copy.deepcopy(coords))


    plot_graph(G)


G = create_graph(56, seed=1)


print(G.starting_node)
print(G.radius)

plot_graph(G)

solve_salesman(G, G.radius, G.starting_node)
