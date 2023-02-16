#!/usr/bin/env python3
"""Calculate autorelation given a network (format is the list of vertice (_vs)
and edges (_es)
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
from myutils import info, create_readme, transform
import igraph
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import product, combinations
from myutils import parallelize
import json
import shutil
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

CID = 'compid'
PALETTE = ['#4daf4a', '#e41a1c', '#ff7f00', '#984ea3', '#ffff33', '#a65628', '#377eb8']

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
        c = coincidence(data2, alpha, coincexp)
        adj[comb[0], comb[1]] = adj[comb[1], comb[0]] = c

    return adj

##########################################################
def get_reachable_vertices_exact(adj, vs0, h):
    """Get the vertices reachable in *exactly* h steps. This
    implies that, for instance, self may be included in the result."""
    if h == 0: return vs0, adj

    adjh = adj
    for i in range(h-1):
        adjh = np.dot(adjh, adj)

    rows, cols = adjh.nonzero()
    reachable = []
    for v in vs0:
        z = cols[np.where(rows == v)]
        reachable.extend(z)

    return reachable, adjh

##########################################################
def get_neighbourhood(adj, vs0, h, itself=False):
    """Get the entire neighbourhood centered on vs0, including self"""
    if h == 0: return vs0 if itself else []
    neighsprev, _ = get_reachable_vertices_exact(adj, vs0, h - 1)
    neighs, _ =  get_reachable_vertices_exact(adj, vs0, h)
    diff = set(neighsprev).union(set(neighs))
    if itself: return diff
    else: return diff.difference(set(vs0))

##########################################################
def get_ring(adj, vs0, h):
    """Get the hth rings"""
    if h == 0: return []
    neigh1 = get_neighbourhood(adj, vs0, h-1)
    neigh2 = get_neighbourhood(adj, vs0, h)
    return list(set(neigh2).difference(set(neigh1)))

##########################################################
def calculate_hiennodes(neighvs): # OK
    return len(neighvs)

##########################################################
def calculate_hienedges(adj, ringcur):
    return adj[ringcur, :][:, ringcur].sum() / 2

##########################################################
def calculate_hierdegree(adj, ringcur, ringnxt):
    return adj[ringcur, :][:, ringnxt].sum()

##########################################################
def calculate_hierclucoeff(he, hn):
    if hn == 1: return 0
    return 2 * (he / (hn * (hn - 1)))

##########################################################
def calculate_hieconvratio(hd, hnnxt):
    if hnnxt == 0: return 0
    return hd / hnnxt

##########################################################
def extract_hirarchical_feats(adj, v, h):
    """Extract hierarchical features"""
    ringcur = get_ring(adj, [v], h)
    ringnxt = get_ring(adj, [v], h+1)
    hn = calculate_hiennodes(ringcur)
    he = calculate_hienedges(adj, ringcur)
    hd = calculate_hierdegree(adj, ringcur, ringnxt)
    hc = calculate_hierclucoeff(he, hn)
    cr = calculate_hieconvratio(hd, calculate_hiennodes(ringnxt))
    return [hn, he, hd, hc, cr]

##########################################################
def extract_hierarchical_feats_all(adj,  h):
    # info(inspect.stack()[0][3] + '()')
    labels = 'hn he hd hc cr'.split(' ')
    feats = []
    for v in range(adj.shape[0]):
        feats.append(extract_hirarchical_feats(adj, v, h))
    return feats, labels

##########################################################
def extract_simple_feats_all(adj, g, h):
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
def generate_graph(model, n, k, outdir):
    """Generate an undirected graph according to @modelstr. It should be MODEL,N,PARAM"""
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

    # gpath = pjoin(outdir, 'graph.png')
    coords = g.layout(layout='fr')
    return g, g.get_adjacency_sparse()

##########################################################
def extract_features(adj, g, h):
    # vfeats, labels = extract_hierarchical_feats_all(adj,  h)
    vfeats, labels = extract_simple_feats_all(adj,  g, h)
    return np.array(vfeats), labels

#############################################################
def get_rgg_params(nvertices, avgdegree):
    rggcatalog = {
        '20000,6': 0.056865545,
    }

    if '{},{}'.format(nvertices, avgdegree) in rggcatalog.keys():
        return rggcatalog['{},{}'.format(nvertices, avgdegree)]

    def f(r):
        g = igraph.Graph.GRG(nvertices, r)
        return np.mean(g.degree()) - avgdegree

    r = scipy.optimize.brentq(f, 0.0001, 10000)
    return r

##########################################################
def plot_graph(g, coordsin, labels, vsizes, outpath):
    coords = np.array(g.layout(layout='fr')) if coordsin is None else coordsin
    visual_style = {}
    visual_style["layout"] = coords
    visual_style["bbox"] = (960, 960)
    visual_style["margin"] = 10
    visual_style['vertex_label'] = labels
    visual_style['vertex_color'] = 'blue'
    visual_style['vertex_size'] = vsizes
    visual_style['vertex_frame_width'] = 0
    igraph.plot(g, outpath, **visual_style)
    return coords

##########################################################
def plot_graph3(g, coordsin, labels, vsizes, grps, outpath):
    coords = np.array(g.layout(layout='fr')) if coordsin is None else coordsin
    visual_style = {}
    visual_style["layout"] = coords
    visual_style["bbox"] = (960, 960)
    visual_style["margin"] = 10
    visual_style['vertex_label'] = labels

    # colors = [ '#1b9e77','#d95f02', '#7570b3', '#e7298a']
    colors = PALETTE
    clrs = []
    for i, grp in enumerate(grps):
        clrs.append(colors[grp])

    visual_style['vertex_color'] = clrs
    visual_style['vertex_size'] = 10
    visual_style['vertex_frame_width'] = 0
    igraph.plot(g, outpath, **visual_style)
    return coords

##########################################################
def plot_graph4(g, coordsin, labels, vsizes, vcolours, outpath):
    coords = np.array(g.layout(layout='fr')) if coordsin is None else coordsin
    visual_style = {}
    visual_style["layout"] = coords
    visual_style["bbox"] = (960, 960)
    visual_style["margin"] = 10
    visual_style['vertex_label'] = labels

    # colors = [ '#1b9e77','#d95f02', '#7570b3', '#e7298a']
    # colors = PALETTE
    # clrs = []
    # for i, grp in enumerate(grps):
        # clrs.append(colors[grp])

    visual_style['vertex_color'] = vcolours
    visual_style['vertex_size'] = 10
    visual_style['vertex_frame_width'] = 0
    igraph.plot(g, outpath, **visual_style)
    return coords

##########################################################
def plot_graph2(g, coordsin, labels, vsizes, mincompsz, outpath):
    coords = np.array(g.layout(layout='fr')) if coordsin is None else coordsin

    membs = np.array(g.vs[CID])
    comps, compszs = np.unique(membs, return_counts=True)
    vcolours = np.array(['#D9D9D9FF'] * g.vcount())
    usedvs = np.where(np.isin(membs, np.where(compszs > mincompsz)[0]))
    vcolours[usedvs] = '#FF0000FF'

    labels = np.array([''] * g.vcount(), dtype=object)
    ##########################################################
    for compid in comps:
        vs = g.vs.select(compid_eq=compid)
        if len(vs) <= mincompsz: continue
        sz = len(vs)
        degs = vs.degree()
        labels[vs.indices[0]] = '{:.02f}'.format(np.mean(degs))
    ##########################################################

    visual_style = {}
    visual_style["layout"] = coords
    visual_style["bbox"] = (1960, 1960)
    visual_style["margin"] = 10
    visual_style['vertex_label'] = labels
    # visual_style['vertex_color'] = 'blue'
    visual_style['vertex_color'] = vcolours
    visual_style['vertex_size'] = vsizes
    visual_style['vertex_frame_width'] = 0
    igraph.plot(g, outpath, **visual_style)
    igraph.plot(g, outpath.replace('.png', '.pdf'), **visual_style)
    return coords

##########################################################
def plot_graph_adj(adj, coords, labels, vsizes, outpath):
    g = igraph.Graph.Weighted_Adjacency(adj, mode='undirected', attr='weight',
                                        loops=False)
    coords = plot_graph(g, coords, labels, vsizes, outpath)
    return coords

##########################################################
def threshold_values(coinc, thresh, newval=0):
    """Values less than or equal to @thresh are set to zero"""
    coinc[coinc <= thresh] = newval
    return coinc

##########################################################
def label_communities(g, attrib, vszs, plotpath):
    vclust = g.components(mode='weak')
    ncomms = vclust.__len__()
    g.vs[CID] = vclust.membership

    membstr = [str(x) for x in g.vs[CID]]
    _ = plot_graph(g, None, None, vszs, plotpath)
    # info(vclust.summary())
    return g

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
def run_experiment(top, nreq, k, h, runid, mincompsz, coincexp, isext, outdir):
    t = 0.8

    random.seed(runid); np.random.seed(runid) # Random seed

    if isext:
        name = os.path.basename(top).replace('.graphml', '')
        name = name.replace('_es.tsv', '')
    else:
        name = top
    expidstr = '{}_{}_{}_{}_{:03d}'.format(name, nreq, k, h, runid)
    info(expidstr)
    tmpfile = pjoin('/tmp/del.png')


    visdir = pjoin(outdir, 'vis')
    os.makedirs(visdir, exist_ok=True)
    op = {'grorig': pjoin(visdir, '{}_grorig.png'.format(expidstr)),
          'grcoinc': pjoin(visdir, '{}_grcoinc.png'.format(expidstr))}

    g, adj = generate_graph(top, nreq, k, outdir)
    n = g.vcount()
    info('n,k:{},{}'.format(n, np.mean(g.degree())))
    vszs = np.array(g.degree()) + 1 # In case it is zero

    # vlbls = [str(i) for i in range(g.vcount())]
    vlbls = None

    coords1 = plot_graph(g, None, vlbls, vszs, tmpfile) # It is going to be overwriten
    vfeats, featlbls = extract_features(adj, g, h)
    coinc = get_coincidx_values(vfeats, .5, coincexp, False)

    # maxdist = g.diameter() + 1
    maxdist = 15

    # For each vx, calculate autorelation across neighbours
    means = np.zeros((n, maxdist), dtype=float)
    stds = np.zeros((n, maxdist), dtype=float)
    for v in range(n):
        dists = np.array(g.distances(source=v, mode='all')[0])
        dists[v] = 9999999 # By default, in a simple graph, a node is not a neighbour of itself
        for l in range(1, maxdist):
            neighs = np.where(dists == l)[0]
            if len(neighs) == 0: continue
            aux = coinc[v, neighs]
            means[v, l], stds[v, l]  = np.mean(aux), np.std(aux)


    # Use autorelation for neighsize={0:maxdist} as features
    coinc = threshold_values(coinc, t)

    # Plot the autorelation curves (one for each vertex)
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    xs = range(1, maxdist)
    for v in range(n):
        ys = means[v, 1:]
        ax.plot(xs, ys)
    ys = np.mean(means[:, 1:], axis=0) # Average of the means
    ax.plot(xs, ys, color='k')
    outpath = pjoin(visdir, '{}_autorel.png'.format(expidstr))
    plt.savefig(outpath); plt.close()
    ys1 = ys

    coords2 = plot_graph_adj(coinc, None, vlbls, vszs, op['grcoinc'])
    gcoinc = igraph.Graph.Weighted_Adjacency(coinc, mode='undirected')
    gcoinc = label_communities(gcoinc, CID, vszs, op['grcoinc'])
    # _ = plot_graph2(gcoinc, None, None, vszs, mincompsz, op['grcoinc']) # Overwrite

    g.vs[CID] = gcoinc.vs[CID]
    membstr = [str(x) for x in g.vs[CID]]

    # _ = plot_graph(g, coords1, vlbls, vszs, tmpfile)
    nclusters = 4
    z = hierarchy.ward(means[:, 1:])
    grps = hierarchy.cut_tree(z, n_clusters=nclusters).flatten()

    scipy.cluster.hierarchy.set_link_color_palette(PALETTE)
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    dendr = hierarchy.dendrogram(z, color_threshold=3, ax=ax)
    aux = dendr['leaves']

    lcolours = np.array(dendr['leaves_color_list'])

    plt.savefig(pjoin(visdir, '{}_dendr.png'.format(expidstr)))

    vcolours = []
    for i in range(n):
        vcolours.append(lcolours[aux.index(i)])

    # vcolours = lcolours[aux]

    coords1 = plot_graph4(g, None, vlbls, vszs, vcolours, op['grorig'])

    ###########################################################
    cols, colcounts =  np.unique(lcolours, return_counts=True)

    x = np.where(np.array(vcolours) == cols[0])[0]
    gind = g.induced_subgraph(x)

    g = gind.copy()
    n = g.vcount()

    maxdist = 15

    # For each vx, calculate autorelation across neighbours
    means = np.zeros((n, maxdist), dtype=float)
    stds = np.zeros((n, maxdist), dtype=float)
    for v in range(n):
        dists = np.array(g.distances(source=v, mode='all')[0])
        dists[v] = 9999999 # By default, in a simple graph, a node is not a neighbour of itself
        for l in range(1, maxdist):
            neighs = np.where(dists == l)[0]
            if len(neighs) == 0: continue
            aux = coinc[v, neighs]
            means[v, l], stds[v, l]  = np.mean(aux), np.std(aux)

    # maxdist = g.diameter() + 1

    # Use autorelation for neighsize={0:maxdist} as features
    coinc = threshold_values(coinc, t)

    # Plot the autorelation curves (one for each vertex)
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    xs = range(1, maxdist)
    for v in range(n):
        ys = means[v, 1:]
        ax.plot(xs, ys)
    ys = np.mean(means[:, 1:], axis=0) # Average of the means
    ax.plot(xs, ys, color='k')
    outpath = pjoin(visdir, '{}_autorel2.png'.format(expidstr))
    plt.savefig(outpath); plt.close()
    
    coords2 = plot_graph_adj(coinc, None, vlbls, vszs, op['grcoinc'])
    gcoinc = igraph.Graph.Weighted_Adjacency(coinc, mode='undirected')
    gcoinc = label_communities(gcoinc, CID, vszs, op['grcoinc'])
    # _ = plot_graph2(gcoinc, None, None, vszs, mincompsz, op['grcoinc']) # Overwrite

    g.vs[CID] = gcoinc.vs[CID]
    membstr = [str(x) for x in g.vs[CID]]

    # _ = plot_graph(g, coords1, vlbls, vszs, tmpfile)
    nclusters = 4
    z = hierarchy.ward(means[:, 1:])
    grps = hierarchy.cut_tree(z, n_clusters=nclusters).flatten()

    scipy.cluster.hierarchy.set_link_color_palette(PALETTE)
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    dendr = hierarchy.dendrogram(z, color_threshold=1.0, ax=ax)
    aux = dendr['leaves']

    lcolours = np.array(dendr['leaves_color_list'])

    plt.savefig(pjoin(visdir, '{}_dendr2.png'.format(expidstr)))

    vcolours = []
    for i in range(n):
        vcolours.append(lcolours[aux.index(i)])

    coords1 = plot_graph4(g, None, vlbls, vszs, vcolours, op['grorig'].replace('.png', '2.png'))
    return ys1

###########################################################
def plot_pca(df, tops, exts, nruns, outdir):
    z = np.diff(df.values, axis=1)
    a, evecs, evals = transform.pca(z, normalize=True)
    # pcs, contribs = transform.get_pc_contribution(evecs)
    coords = np.column_stack([a[:, 0], a[:, 1]]).real

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

    for i in range(len(tops)):
        i0 = i * nruns
        i1 = (i + 1) * nruns
        ax.scatter(coords[i0:i1, 0], coords[i0:i1, 1], label=tops[i], c=PALETTE[i])

    i0 = len(tops) * nruns
    for i, top in enumerate(exts):
        i1 = i0 + i
        lbl = os.path.basename(exts[i]).replace('_es.tsv', '')
        ax.scatter(coords[i1, 0], coords[i1, 1], label=lbl, c=PALETTE[len(tops) + i])
        i1 += 1

    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    outpath = pjoin(outdir, 'pca.png')
    plt.savefig(outpath)

##########################################################
def run_experiments(cfg, nprocs, outpath, outdir):
    tops = cfg['modeltop']
    ns = cfg['modeln']
    ks = cfg['modelk']
    hs = cfg['h']
    coincexp = cfg['coincexp']
    exts = cfg['extmodel']
    mincompsz = 4
    runids = range(cfg['nruns'])
    argsconcat = list(product(tops, ns, ks,  hs, runids, [mincompsz], coincexp, [False], [outdir]))
    args_ = product(exts, [-1], [-1],  hs, [0], [mincompsz], coincexp, [True], [outdir])
    argsconcat.extend(list(args_))
    params = np.array([x[:-1] for x in argsconcat], dtype=object)

    featsall = parallelize(run_experiment, nprocs, argsconcat)
    featsall = np.array(featsall, dtype=object)
    numds = featsall.shape[1]
    featsall = np.column_stack((params, featsall))
    cols1 = ['model', 'nreq', 'k', 'h', 'runid', 'coincexp', 'isext', 'nreal']
    cols2 = ['d{:02d}'.format(d) for d in range(1, numds + 1)]
    cols = cols1 + cols2
    df = pd.DataFrame(featsall.tolist(), columns=cols)
    df.to_csv(outpath, index=False, float_format='%.3f')

    # df = pd.DataFrame(featsall).fillna(0)
    # df.to_csv(outpath, index=False, float_format='%.3f', header=False)
    return df

##########################################################
def main(cfgpath, nprocs, outdir):
    info(inspect.stack()[0][3] + '()')

    cfg = json.load(open(cfgpath))

    outpath = pjoin(outdir, 'res.csv')
    if os.path.isfile(outpath):
        df = pd.read_csv(outpath)
    else:
        df = run_experiments(cfg, nprocs, outpath, outdir)

    plot_pca(df.iloc[:, 8:], cfg['modeltop'], cfg['extmodel'],
             cfg['nruns'], outdir)

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
