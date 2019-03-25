import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def do_node_diffusion(nodes, dx2, D, ts, pd_rate, b):
    """
    nodes a np array of nodes
    dx2, dx2 the difference in x,y squared
    D diffusion constant
    f a method to make a diffusion matrix of available cells
    """
    u = nodes_to_array(nodes).copy()
    array_update_nodes(diffuse(u, dx2, D, pd_rate, b), nodes)
    return nodes


def nodes_to_array(nodes):
    Y, X = nodes.shape
    arr = []
    for y in range(0, Y):
        y_arr = []
        for x in range(0, X):
            y_arr.append(nodes[y, x].get_c())
        arr.append(y_arr)
    return np.array(arr)


def array_update_nodes(arr, nodes):
    Y, X = nodes.shape
    for y in range(0, Y):
        for x in range(0, X):
            nodes[y, x].update_c(arr[y, x])


def array_to_nodes(arr):
    Y, X = arr.shape
    return np.array([[Node(x, y, arr[y, x]) for x in range(0, X)] for y in range(0, Y)])


class Node:
    """
    Is just a holder of data for each node
    """

    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c
        self.closed = False

    def update_c(self, c):
        self.c = c

    def get_c(self):
        return self.c

    def set_closed(self):
        self.closed = True


def array_normalise(arr):
    """
    What percentage of the total sum is each node

    This also assumes no loss...
    """
    return arr/arr.sum()


def draw_as_network(nodes, ax, draw_labels=False, title=''):
    G = nx.Graph()
    Y, X = nodes.shape
    ax.grid(False)
    arr = array_normalise(nodes_to_array(nodes))
    sizes = np.zeros(nodes.shape)
    labels = {}
    log_scale = 1024
    cut_off_of_interest = 1e-4
    pos = {}
    for y in range(0, Y):
        for x in range(0, X):
            cur_node = (y*X) + x
            if x < X-1:
                G.add_edge((cur_node), (cur_node+1),
                           weight=1 if arr[y, x] > cut_off_of_interest else 0)
            if y < Y-1:
                G.add_edge((cur_node), (cur_node+X),
                           weight=1 if arr[y, x] > cut_off_of_interest else 0)

            pos[cur_node] = np.array([x, y])
            sizes[y, x] = arr[y, x]
            lbl = arr[y, x]

            labels[cur_node] = "~{0:.2f}".format(
                np.around(lbl, 4)*100) if lbl > cut_off_of_interest else ''

    with np.errstate(divide='ignore'):
        sizes = sizes.ravel()*10000

    new_sizes = sizes.copy()

    for idx, i in enumerate(G.nodes):
        new_sizes[idx] = sizes[i]

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=new_sizes, ax=ax)

    # Decide nodes
    # e_on = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
    # e_off = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 0]

    nx.draw_networkx_edges(G, pos,
                           width=1, edge_color='b', ax=ax)

    if draw_labels:
        nx.draw_networkx_labels(G, pos, font_size=7,
                                font_family='sans-serif', labels=labels, ax=ax)

    ax.set_xlim(-1, X)
    ax.set_ylim(-1, Y)
    ax.set_title(title)


def to_dataframe(A):
    x1 = np.repeat(np.arange(A.shape[0]), len(
        A.flatten())/len(np.arange(A.shape[0])))
    x2 = np.tile(np.arange(A.shape[1]), int(
        len(A.flatten())/len(np.arange(A.shape[1]))))
    x3 = A.flatten()

    # TODO: Add actual numbers here
    m = np.array([1 for i in range(0, len(x3))])
    # m[3:] = 0
    df = pd.DataFrame(np.array([x1, x2, x3, m]).T,
                      columns=['X', 'Y', 'C', 'M'])
    df['norm_C'] = np.log(df['C'])
    return df
