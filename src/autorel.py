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
def extract_features(adj, g):
    vfeats, labels = extract_simple_feats_all(adj,  g)
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
def plot_graph(g, coordsin, labels, vszs, vcolours, outpath):
    coords = np.array(g.layout(layout='fr')) if coordsin is None else coordsin

    if vszs != None and type(vszs) != int: # Normalize between 5 and 15
        vszs = (vszs - np.min(vszs))/ (np.max(vszs) - np.min(vszs))
        vszs = vszs  * 10 + 5

    visual_style = {}
    visual_style["layout"] = coords
    visual_style["bbox"] = (960, 960)
    visual_style["margin"] = 10
    visual_style['vertex_label'] = labels
    visual_style['vertex_color'] = 'blue' if vcolours == None else vcolours
    visual_style['vertex_size'] = vszs
    visual_style['vertex_frame_width'] = 0

    # widths = np.array(g.es['weight'])
    # widths[widths < 0] = 0
    # visual_style['edge_width'] = np.abs(widths)

    igraph.plot(g, outpath, **visual_style)
    return coords

##########################################################
def threshold_values(coinc, thresh, newval=0):
    """Values less than or equal to @thresh are set to zero"""
    coinc[coinc <= thresh] = newval
    return coinc

##########################################################
def calculate_autorelation(g, coinc, maxdist):
    # For each vx, calculate autorelation across neighbours
    n = g.vcount()
    means = np.zeros((n, maxdist), dtype=float)
    stds = np.zeros((n, maxdist), dtype=float)
    for v in range(n):
        dists = np.array(g.distances(source=v, mode='all')[0])
        dists[v] = 9999999 # In a simple graph, there's no self-loop
        for l in range(1, maxdist):
            neighs = np.where(dists == l)[0]
            if len(neighs) == 0: continue
            aux = coinc[v, neighs]
            means[v, l-1], stds[v, l-1]  = np.mean(aux), np.std(aux)
    return means, stds

##########################################################
def plot_curves_and_avg(curves, ylbl, plotpath):
    """Plot the autorelation curves (one for each vertex)"""
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    maxx = curves.shape[1]
    xs = range(1, maxx + 1)
    for v in range(len(curves)):
        ys = curves[v, :]
        ax.plot(xs, ys)
    ys = np.mean(curves, axis=0) # Average of the means
    ax.plot(xs, ys, color='k')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Shift')
    ax.set_ylabel(ylbl)
    plt.savefig(plotpath); plt.close()

##########################################################
def plot_dendrogram(means, nclusters, expidstr, outdir):
    z = hierarchy.ward(means)
    grps = hierarchy.cut_tree(z, n_clusters=nclusters).flatten()
    uids, counts = np.unique(grps, return_counts=True)
    sortedid = np.argsort(counts)

    colleav = np.array(['#ABB2B9'] * len(grps))
    for i, id in enumerate(sortedid[::-1]):
        inds = np.where(grps == id)[0]
        colleav[inds] = PALETTE[i]

    nlinks = len(z)
    collnks = {}

    for i in range(nlinks):
        clid1, clid2 = np.array(z[i, :2]).astype(int)
        c1 = colleav[clid1] if clid1 <= nlinks else collnks[clid1]
        c2 = colleav[clid2] if clid2 <= nlinks else collnks[clid2]
        collnks[nlinks+i+1] = c1 if c1 == c2 else 'blue'

    def colfunc(id): return collnks[id]

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    dendr = hierarchy.dendrogram(z, link_color_func=colfunc, ax=ax)
    aux = dendr['leaves']
    lcolours = np.array(dendr['leaves_color_list'])
    plt.savefig(pjoin(outdir, '{}_dendr.png'.format(expidstr)))

    vcolours = []
    for i in range(len(means)):
        vcolours.append(lcolours[aux.index(i)])

    return vcolours

##########################################################
def run_experiment(top, n, k, runid, coincexp, outdir):
    """Single run"""
    t = 0.8
    maxdist = 15 # maxdist = g.diameter() + 1
    nclusters = 4  # Hierarchical clustering

    isext = top.endswith('.tsv')
    random.seed(runid); np.random.seed(runid) # Random seed

    gid = os.path.basename(top).replace('_es.tsv', '') if isext else top

    expidstr = '{}_{:02d}'.format(gid, runid)
    info(expidstr)

    visdir = pjoin(outdir, 'vis')
    os.makedirs(visdir, exist_ok=True)

    netorig = pjoin(visdir, '{}_grorig.png'.format(expidstr))
    netcoinc = pjoin(visdir, '{}_grcoinc.png'.format(expidstr))

    g, adj = generate_graph(top, n, k)
    n = g.vcount()
    # info('n,k:{},{:.02f}'.format(n, np.mean(g.degree())))

    vfeats, featlbls = extract_features(adj, g)
    coinc0 = get_coincidx_values(vfeats, .5, coincexp, False)
    means, stds = calculate_autorelation(g, coinc0, maxdist)
    coinc = threshold_values(coinc0, t)
    plotpath = pjoin(visdir, '{}_autorel.png'.format(expidstr))
    plot_curves_and_avg(means, '', plotpath)

    colleav = plot_dendrogram(means, nclusters, expidstr, visdir)

    gcoinc = igraph.Graph.Weighted_Adjacency(coinc, mode='undirected')
    coords1 = plot_graph(gcoinc, None, None, g.vs.degree(), None, netcoinc)
    coords2 = plot_graph(g, None, None, None, colleav, netorig)
    return np.mean(means, axis=0)

###########################################################
def plot_pca(data, models, refmodel, nruns, outdir):
    z = np.diff(data, axis=1)
    a, evecs, evals = transform.pca(z, normalize=True)
    # pcs, contribs = transform.get_pc_contribution(evecs)
    coords = np.column_stack([a[:, 0], a[:, 1]]).real

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

    lbl = os.path.basename(refmodel).replace('_es.tsv', '')
    ax.scatter(coords[0, 0], coords[0, 1], label=lbl,
               c=PALETTE[0])

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

##########################################################
def export_params(tops, n, k, espath, coincexp, nruns, outdir):
    p = {
        'refmodel': espath, 'n': n, 'k': k,
        'coincexp': coincexp, 'nruns': nruns}
    outpath = pjoin(outdir, 'params.json')
    json.dump(p, open(outpath, 'w'))

##########################################################
def run_group(espath, tops, nruns, coincexp, nprocs, outdir):
    """Spawl jobs"""
    outpath = pjoin(outdir, 'res.csv')
    if os.path.isfile(outpath): return pd.read_csv(outpath)
    os.makedirs(outdir, exist_ok=True)

    # Get the n,m,k from the reference network
    g, adj = generate_graph(espath, 0, 0) # connected and undirected
    n, m = [g.vcount()], [g.ecount()]
    k = [m[0] / n[0] * 2]

    export_params(tops, n[0], k[0], espath, coincexp, nruns, outdir)

    runids = range(nruns)
    args1 = [[espath, 0, 0, 0, coincexp, outdir]]
    args2 = list(product(tops, n, k, runids, [coincexp], [outdir]))
    argsconcat = args1 + args2

    avgs = parallelize(run_experiment, nprocs, argsconcat)
    avgs = np.array(avgs)

    # Export averages (black curves)
    nn, mm = avgs.shape
    dfres = pd.DataFrame([[x[0], x[3]] for x in argsconcat],
                         columns=['model', 'runid'])
    for j in range(mm):
        dfres['d{:02d}'.format(j+1)] = avgs[:, j]

    dfres.to_csv(outpath, index=False, float_format='%.3f')

    plot_pca(avgs, tops, espath, nruns, outdir)

##########################################################
def main(cfgpath, nruns, nprocs, outrootdir):
    info(inspect.stack()[0][3] + '()')

    cfg = json.load(open(cfgpath))

    tops = ['ba', 'er', 'ws']
    coincexp = cfg['coincexp'][0]
    netdirs = cfg['netdirs']
    labels = cfg['labels']
    # runids = range(nruns)

    nets, ns, ks = [], [], []
    for d, lbl in zip(netdirs, labels):
        fs = os.listdir(d)
        for f in fs:
            if not f.endswith('_es.tsv'): continue
            fid = f.replace('_es.tsv', '')
            espath = pjoin(d, f)
            outdir = pjoin(outrootdir, lbl, fid)
            run_group(espath, tops, nruns, coincexp, nprocs, outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/toy01.json', help='Experiments settings')
    parser.add_argument('--nruns', default=30, type=int, help='Number of runs for the growth models')
    parser.add_argument('--nprocs', default=1, type=int, help='Number of procs')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)
    shutil.copy(args.config, args.outdir)
    main(args.config, args.nruns, args.nprocs, args.outdir)
    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
