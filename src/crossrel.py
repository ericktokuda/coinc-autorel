#!/usr/bin/env python3
"""Extract features from undirected graph
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

# import warnings
# warnings.filterwarnings("error")

##########################################################
def interiority(dataorig):
    """Calculate the interiority index of the two rows. @vs has 2rows and n-columns, where
    n is the number of features"""
    # info(inspect.stack()[0][3] + '()')
    data = np.abs(dataorig)
    abssum = np.sum(data, axis=1)
    den = np.min(abssum)
    num = np.sum(np.min(data, axis=0))
    # if den == 0:breakpoint()
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
    labels = 'dg cl'.split(' ')
    feats = []
    cl = np.array(g.transitivity_local_undirected(mode=igraph.TRANSITIVITY_ZERO))
    deg = np.array(g.degree()) # np.array(np.sum(adj, axis=0)).flatten()
    feats = np.vstack((deg, cl)).T
    return feats, labels

##########################################################
def vattributes2edges(g, attribs, aggreg='sum'):
    info(inspect.stack()[0][3] + '()')
    m = g.ecount()
    for attrib in attribs:
        values = g.vs[attrib]
        for j in range(m):
            src, tgt = g.es[j].source, g.es[j].target
            if aggreg == 'sum':
                g.es[j][attrib] = g.vs[src][attrib] + g.vs[tgt][attrib]
    return g

##########################################################
def extract_features(adj, g):
    # vfeats, labels = extract_hierarchical_feats_all(adj,  h)
    vfeats, labels = extract_simple_feats_all(adj,  g)
    return np.array(vfeats), labels

##########################################################
def threshold_values(coinc, thresh, newval=0):
    """Values less than or equal to @thresh are set to zero"""
    coinc[coinc <= thresh] = newval
    return coinc

##########################################################
def get_num_adjacent_groups(g, compid):
    membs = g.vs[CID]
    unvisited = set(np.where(np.array(membs) == compid)[0])
    n = len(unvisited)
    visited = set() # Visited but not explored
    explored = set()

    v = unvisited.pop(); visited.add(v)
    adjgrps = []
    group = []

    for i in range(1000000):
        if len(explored) == n: # All vertices from compid was explored
            adjgrps.append(group)
            return n, adjgrps
        elif len(visited) == 0: # No more vertices to explore in this group
            adjgrps.append(group)
            group = []
            v = unvisited.pop(); visited.add(v)
        else:
            v = visited.pop()
            neighs = set(g.neighborhood(v))
            neighs = neighs.intersection(unvisited)
            unvisited = unvisited.difference(neighs)
            visited = visited.union(neighs)
            explored.add(v)
            group.append(v)
    raise Exception('Something wrong')

##########################################################
def get_num_adjacent_groups_all(g):
    membership = g.vs[CID]
    nmembs = len(np.unique(membership))

    nadjgrps = np.zeros((nmembs, 2), dtype=int)
    for compid in range(nmembs):
        compsz, adjgrps = get_num_adjacent_groups(g, compid)
        nadjgrps[compid] = [compsz, len(adjgrps)]
    return nadjgrps

##########################################################
def get_feats_from_components(g, mincompsz):
    FEATLEN = 14
    membs = np.array(g.vs[CID])
    comps, compszs = np.unique(membs, return_counts=True)
    ncomps = len(comps)
    feats = []

    aux = []
    for compid in comps:
        vs = g.vs.select(compid_eq=compid)
        sz = len(vs)
        if sz <= mincompsz: continue
        degs = vs.degree()

        gcomp = g.induced_subgraph(vs)

        dists = np.array(gcomp.distances())
        mpl = np.sum(dists) / (sz * sz - sz)

        clucoeff = gcomp.transitivity_avglocal_undirected()
        aux.append([sz, np.mean(degs), np.std(degs), mpl, clucoeff])

    if len(aux) == 0: return np.array([0] * FEATLEN)

    aux = np.array(aux)
    ws = aux[:, 0] / np.sum(aux[:, 0])
    aux2 = aux[:, 1] * ws
    data = np.column_stack((aux, aux2))

    means = data.mean(axis=0)
    stds = data.std(axis=0)

    feats = [len(data), np.max(data[:, 0])]
    for i in range(data.shape[1]):
        feats.extend([means[i], stds[i]])

    return feats

##########################################################
def get_common_vs(dfvs, id='wid'):
    common = dfvs[0]
    for i in range(1, len(dfvs)):
        common = common.merge(dfvs[i], on='wid')
    return dfvs[0].loc[dfvs[0].wid.isin(common.wid)]

##########################################################
def run_group(f, netdirs, labels, coincexp, nprocs, outdir):
    info(inspect.stack()[0][3] + '()')

    dfes, dfvs, gs = [], [], []
    for d in netdirs:
        dfes.append(pd.read_csv(pjoin(d, f + '_es.tsv'), sep='\t'))
        dfvs.append(pd.read_csv(pjoin(d, f + '_vs.tsv'), sep='\t'))
        gs.append(create_graph_from_dataframes(dfes[-1], dfvs[-1], sep='\t', directed=False))

    dfref = get_common_vs(dfvs)
    dfref.to_csv(pjoin(outdir, '{}_ref.tsv'.format(f)), sep='\t', index=False)

    argsconcat = []
    for i, g in enumerate(gs):
        lbl = '_'.join([f, labels[i]])
        argsconcat.append([g, lbl, dfref, coincexp, outdir])

    featsall = parallelize(run_experiment, nprocs, argsconcat)

##########################################################
def run_experiment(g, lbl, dfref, coincexp, outdir):
    t = 0.8
    info(lbl)

    adj = g.get_adjacency_sparse()
    vfeats, featlbls = extract_features(adj, g)
    coinc = get_coincidx_values(vfeats, .5, coincexp, False)
    nref = len(dfref)

    # maxdist = g.diameter() + 1
    maxdist = 15

    # For each vx, calculate autorelation across neighbours
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

    # Plot the autorelation curves (one for each vertex)
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    xs = range(1, maxdist)
    for v in range(nref):
        ys = means[v, 1:]
        ax.plot(xs, ys)
    ys = np.mean(means[:, 1:], axis=0) # Average of the means
    ax.plot(xs, ys, color='k')
    ax.set_ylim(0, 1)
    outpath = pjoin(outdir, '{}_autorel.png'.format(lbl))
    plt.savefig(outpath); plt.close()

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
def main(cfgpath, nprocs, outdir):
    info(inspect.stack()[0][3] + '()')

    random.seed(0); np.random.seed(0) # Random seed

    cfg = json.load(open(cfgpath))
    respath = pjoin(outdir, 'res.csv')

    coincexp = cfg['coincexp']
    netdirs = cfg['netdirs']
    labels = cfg['labels']

    if len(netdirs) < 2:
        info('There should be at least two folders (cross-relation)')
        return

    suffs = get_suff_of_pairs((find_common_files(netdirs)))

    for suff in suffs:
        run_group(suff, netdirs, labels, coincexp, nprocs, outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/toy01.json', help='Experiments settings')
    parser.add_argument('--nprocs', default=1, type=int, help='Number of procs')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)
    shutil.copy(args.config, args.outdir)
    main(args.config, args.nprocs, args.outdir)
    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
