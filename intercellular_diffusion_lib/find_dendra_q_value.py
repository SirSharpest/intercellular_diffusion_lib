import time
from node_diffusion import do_internode_diffusion
from diffusion_functions import D_eff
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import seaborn as sns
sns.set()

cells = 15  # need to take two extra measurements to get sample of 5 without slicing
cell_um = 100
points_per_cell = 10  # Needs to be divisible by 2 #TODO: put a fix in place
Xs = cells*points_per_cell
u = np.zeros((Xs, 1))
u[(Xs//2)-(points_per_cell//2):(Xs//2)+(points_per_cell//2)] = 1
dx = cell_um//points_per_cell
dx2 = dx**2

b = 0.0
dt = 1

# convert to seconds
t = 1/dt
num_seconds = 60*60*14
ts = int(t * num_seconds)

moss_values = [0.08, 0.2, 0.33, 0.14, 0.05]
stressed_values = [0, 0.12, 0.8, 0.09, 0]
q = 0.0019
step_percent = 0.10


def optimize_mock(q):
    u = np.zeros((Xs, 1))
    u[(Xs//2)-(points_per_cell//2):(Xs//2)+(points_per_cell//2)] = 1
    D = D_eff(83, q, cell_um)
    t1 = time.time()
    nds, u = do_internode_diffusion(
        u, dx2, D, dt, b, cell_um, points_per_cell, ts)
    t2 = time.time()
    print('\n')
    print('took: {0}'.format(t2-t1))
    print('q:{0}'.format(q))
    print('Nodes:{0}'.format(nds))
    print('\n')
    fig = plt.figure(0, figsize=(5, 5))
    fig.clf()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), num=0)
    ax.plot(np.arange(-(cells*cell_um)//2, +(cells*cell_um)//2, step=cell_um)+50,
            nds, marker='o', label='Model')
    ax.set_xlim(-250, 250)
    ax.set_xlabel(r'$\mu m$')
    ax.plot(np.linspace(-200, 200, num=5), moss_values,
            label='Kitawga et al.', marker='o')

    err = np.square((nds[(len(nds)//2)-2:(len(nds)//2)+3] - moss_values)).sum()
    fig.suptitle('Q={0} \t err^2 = {1}'.format(q, err))
    fig.tight_layout()
    fig.savefig('./images/mock_{0}.png'.format(str(q).replace('.', '_')))
    # Minimise sum of squared errors
    return err


def optimize_stress(q):
    u = np.zeros((Xs, 1))
    u[(Xs//2)-(points_per_cell//2):(Xs//2)+(points_per_cell//2)] = 1
    D = D_eff(83, q, cell_um)
    t1 = time.time()
    nds, u = do_internode_diffusion(
        u, dx2, D, dt, b, cell_um, points_per_cell, ts)
    t2 = time.time()
    print('\n')
    print('took: {0}'.format(t2-t1))
    print('q:{0}'.format(q))
    print('Nodes:{0}'.format(nds))
    print('\n')
    fig = plt.figure(0, figsize=(5, 5))
    fig.clf()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), num=0)
    ax.plot(np.arange(-(cells*cell_um)//2, +(cells*cell_um)//2, step=cell_um)+50,
            nds, marker='o', label='Model')
    ax.set_xlim(-250, 250)
    ax.set_xlabel(r'$\mu m$')
    ax.plot(np.linspace(-200, 200, num=5), moss_values,
            label='Kitawga et al.', marker='o')

    err = np.square(
        (nds[(len(nds)//2)-2:(len(nds)//2)+3] - stressed_values)).sum()
    fig.suptitle('Q={0} \t err^2 = {1}'.format(q, err))
    fig.tight_layout()
    fig.savefig('./images/aba_treat_{0}.png'.format(str(q).replace('.', '_')))
    # Minimise sum of squared errors
    return err


result_mock = leastsq(optimize_mock, [q])
result_aba = leastsq(optimize_stress, [q])

print('Mock q: {0} \n ABA q: {1}'.format(result_mock, result_aba))
