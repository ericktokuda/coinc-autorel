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
import myutils
from myutils import info, create_readme, transform, plot
import igraph
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import product, combinations
from myutils import parallelize, graph
import json
import shutil
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
import pickle

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
    info(inspect.stack()[0][3] + '()')
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
def extract_simple_feats_all(adj, g):
    # info(inspect.stack()[0][3] + '()')
    labels = 'dg cl'.split(' ')
    feats = []
    cl = np.array(g.degree()) # Calculating twice the degree instead
    deg = np.array(g.degree())

    # accpath = '/home/dufresne/temp/20230728-accessibs/0671806_Sierra_Madre_undirected_acc05.txt'
    # trans = calculate_trans_feats(g) # Clustering coefficient
    # bet = g.betweenness() # Betweenness centrality
    # accessib = calculate_accessib_feats(accpath) # Accessibility

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
        g = graph.simplify_graphml(model)
        if 'width' in g.edge_attributes(): del g.es['width']
    elif model == 'wx':
        G = nx.waxman_graph(n, beta=0.15, alpha=0.1, L=None, domain=(0, 0, 1, 1), metric=None, seed=None)
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(Gcc[0])
        g = G0
        adj = nx.adjacency_matrix(g)
        vk = dict(G0.degree())
        vk = list(vk.values())
        k = np.mean(vk)
        g = igraph.Graph.from_networkx(G0)
    else:
        raise Exception('Invalid model')

    g.to_undirected()
    g = g.connected_components().giant()
    g.simplify()

    coords = g.layout(layout='fr')
    return g, g.get_adjacency_sparse()

##########################################################
def calculate_trans_feats(g):
    """Calculate clustering coefficient entropy """
    info(inspect.stack()[0][3] + '()')
    clucoeffs = np.array(g.as_undirected().transitivity_local_undirected())

    # TODO: define what do when it is invalid
    # valid = np.argwhere(~np.isnan(clucoeffs)).flatten()
    # clucoeffs = clucoeffs[valid]
    return clucoeffs

##########################################################
def calculate_accessib_feats(accpath):
    """Calculate accessibility features"""
    info(inspect.stack()[0][3] + '()')
    accs = np.loadtxt(accpath)
    return accs

##########################################################
def extract_features(adj, g):
    info(inspect.stack()[0][3] + '()')
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

    a = 7
    b = 3
    if vszs != None and type(vszs) != int: # Normalize between 5 and 15
        vszs = (vszs - np.min(vszs))/ (np.max(vszs) - np.min(vszs))
        vszs = vszs  * a + b

    vszs += 3

    visual_style = {}
    visual_style["layout"] = coords
    visual_style["bbox"] = (1200, 1200)
    visual_style["margin"] = 10
    visual_style['vertex_label'] = labels
    # visual_style['vertex_label'] = range(g.vcount())
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
    info(inspect.stack()[0][3] + '()')
    # For each vx, calculate autorelation across neighbours
    n = g.vcount()
    means = np.zeros((n, maxdist), dtype=float)
    stds = np.zeros((n, maxdist), dtype=float)
    for v in range(n):
        dists = np.array(g.distances(source=v, mode='all')[0])
        dists[v] = 9999999 # In a simple graph, there's no self-loop
        for l in range(1, maxdist + 1):
            neighs = np.where(dists == l)[0]
            if len(neighs) == 0: continue
            aux = coinc[v, neighs]
            means[v, l-1], stds[v, l-1]  = np.mean(aux), np.std(aux)
    return means, stds

##########################################################
def plot_curves_and_avg(curves, ylbl, plotpath):
    """Plot the autorelation curves (one for each vertex)"""
    if isfile(plotpath): return

    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    maxx = curves.shape[1]
    xs = range(1, maxx + 1)

    for v in range(len(curves)): # Plot individual curves
        ys = curves[v, :]
        ax.plot(xs, ys)

        # fig2, ax2 = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        # ax2.plot(xs, ys)
        # ax2.set_ylim(0, 1)
        # ax2.set_xlabel('Shift')
        # ax2.set_ylabel(ylbl)
        # outpath2 = plotpath.replace('.png', '_{:04d}.png'.format(v))
        # plt.savefig(outpath2); plt.close(fig2)
    ys = np.mean(curves, axis=0) # Average of the means
    ax.plot(xs, ys, color='k')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Shift')
    ax.set_ylabel(ylbl)
    plt.savefig(plotpath); plt.close()

##########################################################
def run_experiment(top, n, k, runid, coincexp, maxdist, outrootdir):
    """Single run"""
    info('{} n:{},k:{:.02f}'.format(top, n, k))

    # print(top)
    # if not 'Camarillo' in top: return
    outdir = pjoin(outrootdir, '{:03d}'.format(maxdist))
    dirlayout1 = pjoin(outdir, 'layout1')
    dirlayout2 = pjoin(outdir, 'layout2')
    dirlayout3 = pjoin(outdir, 'layout3')
    os.makedirs(dirlayout1, exist_ok=True)
    os.makedirs(dirlayout2, exist_ok=True)
    os.makedirs(dirlayout3, exist_ok=True)

    isext = top.endswith('.graphml')
    runid += 1
    random.seed(runid); np.random.seed(runid) # Random seed
    gid = os.path.basename(top).replace('.graphml', '') if isext else top
    pklpath = pjoin(outdir, '{}_{:02d}.pkl'.format(gid, runid))

    g, adj = generate_graph(top, n, k)
    info('n:{},k:{:.02f}'.format(g.vcount(), np.mean(g.degree())))

    xy = None
    if 'x' in g.vertex_attributes():
        xy = np.array([g.vs['x'], g.vs['y']]).T
        xy = [[x, -y] for x, y in xy]

    if isfile(pklpath):
        coinc0 = pickle.load(open(pklpath, 'rb'))
    else:
        vfeats, featlbls = extract_features(adj, g)
        coinc0 = get_coincidx_values(vfeats, .5, coincexp, False)
        pickle.dump(coinc0, open(pklpath, 'wb'))

    netorig = pjoin(outdir, '{}_{:02d}.pdf'.format(gid, runid))
    plotpath = pjoin(outdir, '{}_{:02d}_autorel.png'.format(gid, runid))
    coords1 = plot_graph(g, xy, None, 10, None, netorig)

    means, stds = calculate_autorelation(g, coinc0, maxdist)
    plot_curves_and_avg(means, '', plotpath)

    for coincthresh in np.arange(.5, .99, .02):
        expidstr = '{}_T{:.02f}_{:02d}'.format(gid, coincthresh, runid)
        info(expidstr)
        netcoinc1 = pjoin(dirlayout1, '{}.png'.format(expidstr))
        netcoinc2 = pjoin(dirlayout2, '{}.png'.format(expidstr))
        netcoinc3 = pjoin(dirlayout3, '{}.png'.format(expidstr))
        if isfile(netcoinc1) and isfile(netcoinc2): continue

        coinc = threshold_values(coinc0.copy(), coincthresh)
        gcoinc = igraph.Graph.Weighted_Adjacency(coinc, mode='undirected')
        coinc1 = get_coincidx_values(means, .5, coincexp, False)
        coinc = threshold_values(coinc1, coincthresh)
        gcoinc = igraph.Graph.Weighted_Adjacency(coinc, mode='undirected')
        plot_graph(gcoinc, coords1, None, g.vs.degree(), None, netcoinc1)
        plot_graph(gcoinc, None, None, g.vs.degree(), None, netcoinc2)
        giant = gcoinc.components().giant()
        plot_graph(giant, None, None, giant.vs.degree(), None, netcoinc3)

##########################################################
def export_params(tops, n, k, espath, coincexp, nruns, outdir):
    p = {
        'refmodel': espath, 'n': n, 'k': k,
        'coincexp': coincexp, 'nruns': nruns}
    outpath = pjoin(outdir, 'params.json')
    json.dump(p, open(outpath, 'w'))

##########################################################
def run_group(graphml, tops, nruns, coincexp, maxdist, nprocs, readmepath, outdir):
    """Spawl jobs"""
    # Get the n,m,k from the reference network
    g, adj = generate_graph(graphml, 0, 0) # connected and undirected
    ginfo = '{}\tnv:{}\tne:{}'.format(os.path.basename(graphml), g.vcount(),
            g.ecount())
    myutils.append_to_file(readmepath, ginfo)
    n, m = [g.vcount()], [g.ecount()]
    k = [m[0] / n[0] * 2]
    os.makedirs(outdir, exist_ok=True)

    export_params(tops, n[0], k[0], graphml, coincexp, nruns, outdir)

    runids = range(nruns)
    args1 = [[graphml, 0, 0, 0, coincexp, d, outdir] for d in maxdist]
    args2 = list(product(tops, n, k, runids, [coincexp], maxdist, [outdir]))
    argsconcat = args1 + args2

    parallelize(run_experiment, nprocs, argsconcat)

##########################################################
def main(cfgpath, nprocs, readmepath, outrootdir):
    info(inspect.stack()[0][3] + '()')

    cfg = json.load(open(cfgpath))

    tops = []
    coincexp = cfg['coincexp'][0]
    netdirs = cfg['netdirs']
    labels = cfg['labels']
    nruns = cfg['nruns'][0]
    maxdist = cfg['maxdist']

    avgs = {}
    nets, ns, ks = [], [], []
    for d, lbl in zip(netdirs, labels):
        # fs = reversed(os.listdir(d)) # TODO: REMOVE THIS
        fs = os.listdir(d)
        for f in fs:
            if not f.endswith('.graphml'): continue
            fid = f.replace('.graphml', '')
            graphml = pjoin(d, f)
            outdir = pjoin(outrootdir, fid)
            run_group(graphml, tops, nruns, coincexp, maxdist, nprocs,
                    readmepath, outdir)

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
    main(args.config, args.nprocs, readmepath, args.outdir)
    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
