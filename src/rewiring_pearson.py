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
from scipy import stats
import math
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
from myutils import info, create_readme, transform


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
    ax.set_xlabel('Modification index')
    plt.savefig(plotpath); plt.close()
    return sums

##########################################################
def plot_networks(gs, vschg, dfref, diffsums, coords, means, labels, outdir):

    n = gs[0].vcount()
    vclr = ['blue'] * n
    for v in vschg: vclr[v] = 'red'
    # vszs = (diffsums - np.min(diffsums)) / (np.max(diffsums) - np.min(diffsums))
    # vszs = (vszs * 25) + 5
    vszs = np.log(diffsums + .001)
    vszs = (vszs - np.min(vszs)) / (np.max(vszs) - np.min(vszs))
    vszs = (vszs * 20) + 5
    bbox = (600, 600)

    gsind = []
    i = 1 # Just plot the modified network
    plotpath = pjoin(outdir, '{}_netw.png'.format(labels[i]))
    z = [gs[i].vs.find(wid=x).index for x in dfref.wid.values]
    gsind.append(gs[i].induced_subgraph(z))
    # vlbl = [str(x) for x in range(gsind[i].vcount())]
    vlbl = None
    if coords == None: coords = gsind[-1].layout('fr')

    if not isfile(plotpath):
        igraph.plot(gsind[0], plotpath, layout=coords, vertex_size=vszs,
                    vertex_color=vclr, vertex_label=vlbl, bbox=bbox)

    pcapath = pjoin(outdir, 'pca.png')
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    a, evecs, evals = transform.pca(means, normalize=True)
    ax.scatter(a[vschg, 0], a[vschg, 1], label='changed', alpha=.4, c='red')
    nonchg = np.ones(a.shape[0], dtype=bool); nonchg[vschg] = 0
    ax.scatter(a[nonchg, 0], a[nonchg, 1], label='Non changed', alpha=.4, c='blue')
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(pcapath); plt.close()

    meanspath = pjoin(outdir, 'means.png')
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.scatter(means[vschg, 0], means[vschg, 1], label='changed', alpha=.4, c='red')
    nonchg = np.ones(means.shape[0], dtype=bool); nonchg[vschg] = 0
    ax.scatter(means[nonchg, 0], means[nonchg, 1], label='Non changed', alpha=.4, c='blue')
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(meanspath); plt.close()
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
def calculate_pearson(dfref, gs, maxdist):
    """Calculate autorelation of the vertices @dfref in @g"""
    # For each vx, calculate autorelation across neighbours
    nref = len(dfref)
    coeffs = np.zeros((nref, maxdist), dtype=float)

    lengths = []
    g = gs[0]
    means = np.zeros((len(dfref), 2))
    for i, wid in enumerate(dfref.wid):
        v = g.vs.find(wid=wid).index
        dists = np.array(g.distances(source=v, mode='all')[0])
        dists[v] = 9999999 # By default, in a simple graph, a node is not a neighbour of itself
        row = []
        for l in range(1, maxdist):
            neighs = np.where(dists == l)[0]
            if len(neighs) == 0:
                pass
            elif len(neighs) <= 2:
                coeffs[i, l] = 0
            else:
                degs0 = gs[0].vs[neighs].degree()
                degs1 = gs[1].vs[neighs].degree()
                coeff, pval = stats.pearsonr(degs0, degs1)
                coeffs[i, l] = 0 if math.isnan(coeff) else coeff
                mean0, mean1 = np.mean(degs0), np.mean(degs1)
                means[l, :] = [mean0, mean1]

            row.append(len(neighs))
        lengths.append(row)

    # print(means)
    # breakpoint()
    return coeffs, lengths, means

##########################################################
def calculate_coinc2(dfref, gs, maxdist):
    """Calculate autorelation of the vertices @dfref in @g"""
    # For each vx, calculate autorelation across neighbours
    nref = len(dfref)
    coeffs = np.zeros((nref, maxdist), dtype=float)

    lengths = []
    g = gs[0]
    means = np.zeros((len(dfref), 2))
    for i, wid in enumerate(dfref.wid):
        row = []
        v = g.vs.find(wid=wid).index
        dists = np.array(g.distances(source=v, mode='all')[0])
        dists[v] = 9999999 # By default, in a simple graph, a node is not a neighbour of itself
        for l in range(1, maxdist):
            neighs = np.where(dists == l)[0]
            if len(neighs) == 0:
                pass
            elif len(neighs) <= 2:
                coeffs[i, l] = 0
            else:
                degs0 = gs[0].vs[neighs].degree()
                degs1 = gs[1].vs[neighs].degree()
                data2 = np.array([degs0, degs1])
                coeff = coincidence(data2, 0.5, 1.0)

                coeffs[i, l] = 0 if math.isnan(coeff) else coeff
                mean0, mean1 = np.mean(degs0), np.mean(degs1)
                means[l, :] = [mean0, mean1]

            row.append(len(neighs))
        lengths.append(row)
    return coeffs, lengths, means

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
def run_experiment(gs, dfref, simil, maxdist):
    if simil == 'pear': coeffs, lengths, means  = calculate_pearson(dfref, gs, maxdist)
    elif simil == 'coinc': coeffs, lengths, means = calculate_coinc2(dfref, gs, maxdist)
    return coeffs, lengths, means

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
def main(nprocs, outrootdir):
    info(inspect.stack()[0][3] + '()')

    random.seed(0); np.random.seed(0) # Random seed

    maxdist = 25    # Max distance from the central node
    coincexp = 1    # Coincidence exponent
    nruns = 2   # Number of experiments
    nes = 1    # Number of new edges per run
    models = ['ba', 'er']    # Network growth-model
    nreq = 200  # Network requested num vertices
    k = 6   # Network average degree
    q1, q2 = .1, .9 # Quantiles

    for model in models:
        for simil in ['pear', 'coinc']:
            outdir = pjoin(outrootdir, simil)
            os.makedirs(outdir, exist_ok=True)
            run_model(model, nreq, k, q1, q2, maxdist, coincexp, nes, nruns,
                      simil, outdir)

##########################################################
def plot_deg_vs_diffsum(degs, diffsum, xlim, outpath):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.scatter(degs, diffsum, c='blue')
    # ax.scatter(degs, diffsum, c='blue', alpha=.5)
    ax.set_xlabel('Degree')
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Modification index')
    plt.savefig(outpath); plt.close()

###########################################################
def plot_pca(data, models, refmodel, nruns, outdir):
    data = np.diff(data, axis=1) # Notice this line!
    a, evecs, evals = transform.pca(data, normalize=True)

    # pcs, contribs = transform.get_pc_contribution(evecs)
    coords = np.column_stack([a[:, 0], a[:, 1]]).real

    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

    lbl = os.path.basename(refmodel).replace('_es.tsv', '')
    ax.scatter(coords[0, 0], coords[0, 1], label=lbl, c=PALETTE[0])

    coords = coords[1:, :]
    for topid in range(len(models)):
        i0 = topid * nruns
        i1 = (topid + 1) * nruns
        ax.scatter(coords[i0:i1, 0], coords[i0:i1, 1],
                   label=models[topid], c=PALETTE[topid+1])

    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    outpath = pjoin(outdir, 'pca.png')
    plt.savefig(outpath); plt.close()
    return coords

##########################################################
def run_model(model, nreq, k, q1, q2, maxdist, coincexp, nes, nruns,
              simil, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')

    g0, _ = generate_graph(model, nreq, k)
    n0 = g0.vcount()

    degs = np.array(g0.degree())
    inds = np.argsort(degs)
    ids1, ids2 = inds[:int(q1 * n0)], inds[int(q2 * n0):]

    grps = {'q1': ids1, 'q2': ids2}
    ngrps = len(grps)

    g0.vs['wid'] = range(g0.vcount())
    dfref = pd.DataFrame(range(g0.vcount()), columns=['wid'])

    lbl0 = '{}_orig'.format(model)
    xlim = (0, maxdist) # Shifts lim (min and max)
    coords = None
    g1s = []

    for i, q in enumerate(grps.keys()): # For each groups of nodes
        info('grp: {}'.format(q))

        inds1, inds2 = set(grps[q]), set(range(n0))
        inds3 = np.array(list(inds2.difference(inds1)))
        diffsall = []

        for r in range(nruns): # For each run
            info('run: {}/{}'.format(r, nruns))

            lbl1 = '{}_{}_{:03d}'.format(model, q, r)

            # Apply changes to the network
            g1 = g0.copy()
            for j, v in enumerate(grps[q]): # For each source vx
                neighs0 = [x.index for x in g0.vs[v].neighbors()]
                # Non-linked vertices
                vsnlink = list(set(range(n0)).difference(neighs0))
                np.random.shuffle(vsnlink)
                es = [[v, v2] for v2 in vsnlink[:nes]]
                g1.add_edges(es)

            # Extract features and calculate cross relation
            m1, lengths, means = run_experiment([g0, g1], dfref, simil, maxdist)

            plotpath = pjoin(outdir, '{}_count.png'.format(lbl1, q))
            fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
            for row in lengths:
                # ax.plot(range(3, len(row)), row[3:])
                ax.plot(range(len(row)), ([0] * 3 + list(row[3:])))
                ax.set_xlabel('Lag')
                ax.set_ylabel('Count')
            ax.set_xlim(0, len(lengths[0]))
            plt.savefig(plotpath)

            plotpath = pjoin(outdir, '{}_deg_diffsum.png'.format(lbl1, q))
            diffs2 = m1.sum(axis=1)
            plot_deg_vs_diffsum(degs, diffs2, xlim, plotpath)


            coords = plot_networks([g0, g1], grps[q], dfref, diffs2,
                                   coords, means, [lbl0, lbl1], outdir)


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
