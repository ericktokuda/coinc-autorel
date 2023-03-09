#!/usr/bin/env python3
"""Calculate Luc's cross-relation of all graphs that are simultaneously inside a set of folders
"""

import argparse
import time, datetime
import os, sys, random
from os.path import join as pjoin
from os.path import isfile
import inspect

import numpy as np
import scipy; import scipy.optimize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme, transform, parallelize
import igraph
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import product, combinations
import json
import shutil
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from utils import create_graph_from_dataframes

CID = 'compid'
PALETTE = ['#4daf4a', '#e41a1c', '#ff7f00', '#984ea3', '#ffff33', '#a65628', '#377eb8']
W = 640; H = 480

##########################################################
def interiority(dataorig):
    """Calculate the interiority index of the two rows. @vs has 2rows and n-columns, where
    n is the number of features"""
    # info(inspect.stack()[0][3] + '()')
    data = np.abs(dataorig)
    abssum = np.sum(data, axis=1)
    den = np.min(abssum)
    num = np.sum(np.min(data, axis=0))
    return num / den

##########################################################
def jaccard(dataorig, a):
    """Calculate the interiority index of the two rows. @vs has 2rows and n-columns, where
    n is the number of features"""
    data = np.abs(dataorig)
    den = np.sum(np.max(data, axis=0))
    datasign = np.sign(dataorig)
    plus_ = np.abs(datasign[0, :] + datasign[1, :])
    minus_ = np.abs(datasign[0, :] - datasign[1, :])
    splus = np.sum(plus_ * np.min(data, axis=0))
    sminus = np.sum(minus_ * np.min(data, axis=0))
    num = a * splus - (1 - a) * sminus
    return num / den

##########################################################
def coincidence(data, a, D):
    inter = interiority(data)
    jac = jaccard(data, a)
    return inter * np.power(jac, D)

##########################################################
def get_coincidx_values(dataorig, alpha, coincexp, standardize):
    """Get coincidence value between each combination in @dataorig"""
    # info(inspect.stack()[0][3] + '()')
    n, m = dataorig.shape
    if standardize:
        data = StandardScaler().fit_transform(dataorig)
    else:
        data = dataorig

    adj = np.zeros((n, n), dtype=float)
    for comb in list(combinations(range(n), 2)):
        data2 = data[list(comb)]

        if np.all(data2[0, :] != 0) and np.all(data2[1, :] != 0):
            c = coincidence(data2, alpha, coincexp)
        else:
            c = 0

        adj[comb[0], comb[1]] = adj[comb[1], comb[0]] = c

    return adj

##########################################################
def extract_simple_feats_all(adj, g):
    # info(inspect.stack()[0][3] + '()')
    # labels = 'dg cl'.split(' ')
    labels = 'dg dg'.split(' ')
    feats = []
    # cl = np.array(g.transitivity_local_undirected(mode=igraph.TRANSITIVITY_ZERO))
    deg = np.array(g.degree()) # np.array(np.sum(adj, axis=0)).flatten()
    cl = np.array(g.degree()) # np.array(np.sum(adj, axis=0)).flatten()
    feats = np.vstack((deg, cl)).T
    return feats, labels

##########################################################
def plot_abs_diff(feat1, feat2, lbl, outdir):
    diffs = np.abs(feat1 - feat2)
    plotpath = pjoin(outdir, '{}_diff.png'.format(lbl))
    plot_curves_and_avg(diffs, '', plotpath)
    return diffs

##########################################################
def plot_diff_sums_hist(diffs, lbl, outdir):
    sums = np.sum(diffs, axis=1)
    plotpath = pjoin(outdir, '{}_diffsum.png'.format(lbl))
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.hist(sums, bins=50)
    ax.set_xlabel('Sum of crossrelation across the neighbours of each node')
    plt.savefig(plotpath); plt.close()
    return sums

##########################################################
def plot_networks(gs, vschg, dfref, diffsums, coords, labels, outdir):

    n = gs[0].vcount()
    vclr = ['blue'] * n
    for v in vschg: vclr[v] = 'red'
    # vszs = (diffsums - np.min(diffsums)) / (np.max(diffsums) - np.min(diffsums))
    # vszs = (vszs * 25) + 5
    vszs = np.log(diffsums + .001)
    vszs = (vszs - np.min(vszs)) / (np.max(vszs) - np.min(vszs))
    vszs = (vszs * 25) + 5
    bbox = (600, 600)

    gsind = []
    for i in range(2):
        z = [gs[i].vs.find(wid=x).index for x in dfref.wid.values]
        gsind.append(gs[i].induced_subgraph(z))
        # vlbl = [str(x) for x in range(gsind[i].vcount())]
        vlbl = None
        if coords == None: coords = gsind[-1].layout('fr')
        plotpath = pjoin(outdir, '{}_netw.png'.format(labels[i]))
        igraph.plot(gsind[i], plotpath, layout=coords, vertex_size=vszs, vertex_color=vclr,
                    vertex_label=vlbl, bbox=bbox)
    return coords

##########################################################
def calculate_crossrelation(dfref, g, coinc, maxdist):
    """Calculate autorelation of the vertices @dfref in @g"""
    # For each vx, calculate autorelation across neighbours
    nref = len(dfref)
    means = np.zeros((nref, maxdist), dtype=float)
    stds = np.zeros((nref, maxdist), dtype=float)
    for i, wid in enumerate(dfref.wid):
        v = g.vs.find(wid=wid).index
        dists = np.array(g.distances(source=v, mode='all')[0])
        dists[v] = 9999999 # By default, in a simple graph, a node is not a neighbour of itself
        for l in range(1, maxdist):
            neighs = np.where(dists == l)[0]
            if len(neighs) == 0: continue
            aux = coinc[v, neighs]
            means[i, l], stds[i, l]  = np.mean(aux), np.std(aux)
    return means, stds

##########################################################
def plot_crossrelation(means, stds, lbl, outdir):
    plotpath = pjoin(outdir, '{}_crossrel.png'.format(lbl))
    plot_curves_and_avg(means, '', plotpath)

##########################################################
def plot_curves_and_avg(curves, ylbl, plotpath):
    """Plot the autorelation curves (one for each vertex)"""
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    maxx = curves.shape[1]
    xs = range(1, maxx + 1)
    for v in range(len(curves)):
        ys = curves[v, :]
        ax.plot(xs, ys)
    ys = np.mean(curves, axis=0) # Average of the means
    ax.plot(xs, ys, color='k')
    # ax.set_ylim(0, 1)
    ax.set_xlabel('Shift')
    ax.set_ylabel(ylbl)
    plt.savefig(plotpath); plt.close()

##########################################################
def run_experiment(g, lbl, dfref, maxdist, coincexp, outdir):
    adj = g.get_adjacency_sparse()
    vfeats, featlbls = extract_simple_feats_all(adj,  g)
    vfeats = np.array(vfeats)
    coinc = get_coincidx_values(vfeats, .5, coincexp, False)
    means, stds = calculate_crossrelation(dfref, g, coinc, maxdist)
    plot_crossrelation(means[:, 1:], stds[:, 1:], lbl, outdir)
    return [means, stds]

##########################################################
def find_common_files(dirs, ext='.tsv'):
    fs = set(os.listdir(dirs[0]))
    for d in dirs[1:]:
        fs = fs.intersection(os.listdir(d))
    return sorted(list(fs))

##########################################################
def get_suff_of_pairs(fs):
    filtered = []
    for f in fs:
        suff = f.replace('_es.tsv', '').replace('_vs.tsv', '')
        fes, fvs = suff + '_es.tsv', suff + '_vs.tsv'
        if (fes in fs) and (fvs in fs): filtered.append(suff)
    return filtered

##########################################################
def generate_graph(model, n, k):
    """Generate an undirected connected graph according to @modelstr. It should be MODEL,N,PARAM"""
    # info(inspect.stack()[0][3] + '()')

    if model == 'er':
        erdosprob = k / n
        if erdosprob > 1: erdosprob = 1
        g = igraph.Graph.Erdos_Renyi(n, erdosprob)
    elif model == 'ba':
        m = round(k / 2)
        if m == 0: m = 1
        g = igraph.Graph.Barabasi(n, m)
    elif model == 'gr':
        r = get_rgg_params(n, k)
        g = igraph.Graph.GRG(n, radius=r, torus=False)
    elif model == 'ws':
        rewprob = .05
        g = igraph.Graph.Watts_Strogatz(1, n, 2, p=rewprob)
    elif model.endswith('.graphml'):
        g = igraph.Graph.Read_GraphML(model)
        g.vs['wid'] = [int(x) for x in g.vs['wid']]
        del g.vs['id'] # From graphml
    elif model.endswith('_es.tsv'):
        dfes = pd.read_csv(model, sep='\t')
        dfvs = pd.read_csv(model.replace('_es.tsv', '_vs.tsv'), sep='\t')
        nreal = len(dfvs)
        g = igraph.Graph(nreal)
        g.add_edges(dfes.values)
        for c in dfvs.columns: g.vs[c] = dfvs[c]
    elif model == 'sb':
        if n == 200 and k == 6: p1 = .07
        elif n == 350 and k == 6: p1 = .04
        elif n == 500 and k == 6: p1 = .0279
        elif n == 200 and k == 12: p1 = .0798
        elif n == 200 and k == 18: p1 = .1197
        else: p1 = .1

        p2 = p1 / 10
        pref = np.diag([p1, p1, p1]) + p2
        n1 = int(n / 3)
        szs = [n1, n1, n - 2 * n1]
        g = igraph.Graph.SBM(n, pref.tolist(), szs, directed=False, loops=False)
    else:
        raise Exception('Invalid model')

    g.to_undirected()
    g = g.connected_components().giant()
    g.simplify()

    coords = g.layout(layout='fr')
    return g, g.get_adjacency_sparse()

##########################################################
def main(nprocs, outdir):
    info(inspect.stack()[0][3] + '()')

    random.seed(0); np.random.seed(0) # Random seed

    maxdist = 15    # Max distance from the central node
    coincexp = 1    # Coincidence exponent
    nruns = 3   # Number of experiments
    nes = 1    # Number of new edges per run
    models = ['ba', 'er']    # Network growth-model
    nreq = 200  # Network requested num vertices
    k = 6   # Network average degree
    q1, q2 = .1, .9 # Quantiles

    for model in models:
        run_model(model, nreq, k, q1, q2, maxdist, coincexp, nes, nruns,
                  pjoin(outdir, model))

##########################################################
def plot_deg_vs_diffsum(degs, diffsum, xlim, outpath):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.scatter(degs, diffsum)
    ax.set_xlabel('Degree')
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Sum of absolute difference')
    plt.savefig(outpath)

##########################################################
def run_model(model, nreq, k, q1, q2, maxdist, coincexp, nes, nruns,
              outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    os.makedirs(outdir, exist_ok=True)
    g0, _ = generate_graph(model, nreq, k)
    n0 = g0.vcount()

    degs = np.array(g0.degree())
    inds = np.argsort(degs)
    ids1, ids2 = inds[:int(q1 * n0)], inds[int(q2 * n0):]

    grps = {'q1': ids1, 'q2': ids2}
    ngrps = len(grps)

    g0.vs['wid'] = range(g0.vcount())
    dfref = pd.DataFrame(range(g0.vcount()), columns=['wid'])

    lbl0 = 'orig'
    mean0, std0 = run_experiment(g0, lbl0, dfref, maxdist, coincexp,
                                 outdir)

    xlim = (0, 25) # Shifts lim (min and max)
    coords = None
    g1s = []
    esall = {}
    for i, q in enumerate(grps.keys()): # For each groups of nodes
        info('grp: {}'.format(q))
        esall[q] = []

        for r in range(nruns): # For each run
            info('run: {}/{}'.format(r, nruns))

            lbl1 = '{}_{:03d}'.format(q, r)

            # Apply changes to the network
            g1 = g0.copy()
            for j, v in enumerate(grps[q]): # For each source vx
                neighs0 = [x.index for x in g0.vs[v].neighbors()]
                # Non-linked vertices
                vsnlink = list(set(range(n0)).difference(neighs0))
                np.random.shuffle(vsnlink)
                es = [[v, v2] for v2 in vsnlink[:nes]]
                g1.add_edges(es)
                esall[q].extend(es)
            g1s.append(g1)

            # Extract features and calculate cross relation
            m1, s1 = run_experiment(g1, lbl1, dfref, maxdist,
                                    coincexp, outdir)

            mean1, std1 = np.mean(m1), np.mean(s1)

            # Plot differences
            diffs = plot_abs_diff(mean0, mean1, lbl1, outdir)
            diffsums = plot_diff_sums_hist(diffs, lbl1, outdir)

            plotpath = pjoin(outdir, '{}_deg_diffsum.png'.format(lbl1, q))
            plot_deg_vs_diffsum(degs, diffsums, xlim, plotpath)

            inds1, inds2 = set(grps['q1']), set(range(n0))
            inds3 = np.array(list(inds2.difference(inds1)))

            plotpath = pjoin(outdir, '{}_deg_diffsum2.png'.format(lbl1, q))
            plot_deg_vs_diffsum(degs[inds3], diffsums[inds3], xlim, plotpath)

            coords = plot_networks([g0, g1s[i]], grps[q], dfref, diffsums,
                                   coords, [lbl0, lbl1], outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nprocs', default=1, type=int, help='Number of procs')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)
    main(args.nprocs, args.outdir)
    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
