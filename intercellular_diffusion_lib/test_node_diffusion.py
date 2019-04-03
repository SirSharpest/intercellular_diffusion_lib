from node_diffusion import do_internode_diffusion, draw_as_network, array_to_nodes
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(0, figsize=(5, 5))
fig.clf()
fig, ax = plt.subplots(1, 1, figsize=(5, 5), num=0)


def test_as_imgs():
    fig = plt.figure(0, figsize=(5, 5))
    fig.clf()
    fig, ax = plt.subplots(1, 2, figsize=(5, 5), num=0)

    nds, u = do_internode_diffusion(u, dx2, D, dt, b, points_per_cell, ts)

    [ax[0].axhline(y=i, color="red")
     for i in np.arange(0, cells*points_per_cell, step=points_per_cell)]
    [ax[0].axvline(x=i, color="red")
     for i in np.arange(0, cells*points_per_cell, step=points_per_cell)]

    [ax[1].axhline(y=i, color="red")
     for i in np.arange(-0.5, cells, step=1)]
    [ax[1].axvline(x=i, color="red")
     for i in np.arange(-0.5, cells, step=1)]

    ax[0].imshow(u)
    ax[1].imshow(nds)


cells = 3  # Per dimension
cell_um = 100
points_per_cell = 10
Xs = cells*points_per_cell
Ys = cells*points_per_cell
u = np.zeros((Xs, Ys))
u[(Ys//2)-(points_per_cell//2):(Ys//2)+(points_per_cell//2),
  (Xs//2)-(points_per_cell//2):(Xs//2)+(points_per_cell//2)] = 1
dx = cell_um//points_per_cell
dx2 = dx**2
b = 0.0
dt = .01
# convert to seconds
t = 1/dt
num_seconds = 1
ts = int(t * num_seconds)
D = 300

nds, u = do_internode_diffusion(u, dx2, D, dt, b, points_per_cell, ts)
draw_as_network(array_to_nodes(nds), ax, draw_labels=True)

fig.tight_layout()
fig.canvas.draw()
