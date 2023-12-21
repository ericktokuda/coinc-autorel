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
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from itertools import product, combinations
from myutils import parallelize, graph
import json
import shutil
import pickle

# PALETTE = ['#4daf4a', '#e41a1c', '#ff7f00', '#984ea3', '#ffff133', '#a65628', '#377eb8']
PALETTE = ['#FF0000','#0000FF','#00FF00','#8e00a3','#ff7f00','#ffff33','#a65628','#f70082']
W = 640; H = 480
R = 6371 # Earth radius

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
def get_coincidx_values(dataorig, coincexp, standardize):
    """Get coincidence value between each combination in @dataorig"""
    info(inspect.stack()[0][3] + '()')
    alpha = .5
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
def get_farthest_points(pos, metric='euclidean'):
    """Get farthest points and corresponding distance (diameter)."""
    info(inspect.stack()[0][3] + '()')
    diam = -1
    i0, j0 = -1, -1
    for i in range(len(pos) - 1):
        p1 = [g.vs[i]['x'], g.vs[i]['y']]
        for j in range(i + 1, len(pos)):
            p2 = [g.vs[j]['x'],g.vs[j]['y']]
            dist = myutils.geo.haversine(p1, p2)
            if dist > diam:
                diam = dist; i0 = i; j0 = j
    return diam, [i0, j0]

##########################################################
def extract_simple_feats_all(adj, g, isgeo):
    # info(inspect.stack()[0][3] + '()')
    labels = 'dg ds'.split(' ')
    feats = []
    # cl = np.array(g.degree()) # Calculating twice the degree instead
    deg = np.array(g.degree())

    # accpath = '/home/dufresne/temp/20230728-accessibs/0671806_Sierra_Madre_undirected_acc05.txt'
    ##########################################################

    factor = 15
    if isgeo:
        pos = np.array([g.vs['y'], g.vs['x']]).T
        posdeg = np.deg2rad(pos)
        dists = sklearn.metrics.pairwise.haversine_distances(posdeg)
        diamkm = (np.max(dists) * R)
        radkm = diamkm / factor
        # radkm = .5 # TODO: Remove this
        info('Using a radius of :{}'.format(radkm))
        bt = sklearn.neighbors.BallTree(posdeg, metric='haversine')
        counts = bt.query_radius(posdeg, r=radkm/R, count_only=True)
        g['featrad'] = radkm
    else:
        pos = np.array([g.vs['x'], g.vs['y']]).T
        dists = sklearn.metrics.pairwise.euclidean_distances(pos)
        diam = np.max(dists)
        rad = diam / factor
        bt = sklearn.neighbors.BallTree(pos, metric='euclidean')
        counts = bt.query_radius(pos, r=rad, count_only=True)
        g['featrad'] = rad

    # clucoeffs = np.array(g.as_undirected().transitivity_local_undirected(mode='zero'))

    # bet = g.betweenness() # Betweenness centrality
    # accessib = calculate_accessib_feats(accpath) # Accessibility

    feats = np.vstack((deg, counts)).T
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
        # g = g.induced_subgraph(range(800)).components().giant() # TODO: remove this
        if 'width' in g.edge_attributes(): del g.es['width']
        # model = os.path.basename(model)
    elif model == 'wx':
        G = nx.waxman_graph(n, beta=0.15, alpha=0.1, L=None, domain=(0, 0, 1, 1), metric=None, seed=None)
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(Gcc[0])
        g = G0
        vk = dict(G0.degree())
        vk = list(vk.values())
        k = np.mean(vk)
        g = igraph.Graph.from_networkx(G0)
    elif model == 'sbm':
        nblocks = 4
        blocksz = n // nblocks
        probloose = .005
        prefmatrix = np.ones((nblocks, nblocks), dtype=float) * probloose
        prefmatrix[0 ,0] = .40
        prefmatrix[1 ,1] = .30
        prefmatrix[2 ,2] = .20
        prefmatrix[3 ,3] = .10
        blockszs = [blocksz, blocksz, blocksz, n - (nblocks - 1) * blocksz]

        grps = []
        for i in range(nblocks):
            grps += [i] * blocksz # Be careful with this list multipl operation

        g = igraph.Graph.SBM(n, list(prefmatrix), blockszs, directed=False, loops=False)
    else:
        raise Exception('Invalid model')

    g.to_undirected()
    g = g.connected_components().giant()
    g.simplify()

    coords = g.layout(layout='fr')
    g['model'] = model
    return g, g.get_adjacency_sparse()

##########################################################
def calculate_accessib_feats(accpath):
    """Calculate accessibility features"""
    info(inspect.stack()[0][3] + '()')
    accs = np.loadtxt(accpath)
    return accs

##########################################################
def extract_features(adj, g, isgeo):
    info(inspect.stack()[0][3] + '()')
    vfeats, labels = extract_simple_feats_all(adj,  g, isgeo)
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
def plot_graph(g, coordsin, labels, vszs, vcolours, outpath, bbox=(1600, 1600)):
    coords = np.array(g.layout(layout='fr')) if coordsin is None else coordsin

    a = 7
    b = 3
    if vszs != None and type(vszs) != int: # Normalize between 5 and 15
        vszs = (vszs - np.min(vszs))/ (np.max(vszs) - np.min(vszs))
        vszs = vszs  * a + b

    vszs += 3

    alpha = .4 - (2.3e-6) * g.ecount() # Magical numbers calibrated to our networks
    if alpha < 0.1:  alpha = .1
    elif alpha > 1: alpha = 1
    alphahex = hex(int(alpha * 255))

    visual_style = {}
    visual_style["layout"] = coords
    visual_style["bbox"] = bbox
    visual_style["margin"] = 10
    visual_style['vertex_label'] = labels
    # visual_style['vertex_label'] = range(g.vcount())
    visual_style['vertex_color'] = 'blue' if vcolours == None else vcolours
    visual_style['vertex_size'] = vszs
    visual_style['vertex_frame_width'] = 1
    visual_style['vertex_frame_width'] = 1
    visual_style['edge_color'] = '#000000' + alphahex[-2:]

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
def plot_curves_and_avg_comm(curves, ylbl, vcols, lbl, outdir):
    """Plot the autorelation curves (one for each vertex)"""
    maxx = curves.shape[1]
    xs = range(1, maxx + 1)

    vcolsuniq, counts = np.unique(vcols, return_counts=True)
    vcolsuniq = vcolsuniq[np.argsort(counts)][::-1] # Ordered by comm size

    figs, axs, yss = {}, {}, {}
    for col in vcolsuniq:
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        figs[col] = fig
        axs[col] = ax
        yss[col] = []

    for v in range(len(curves)): # Plot individual curves
        ys = curves[v, :]
        axs[vcols[v]].plot(xs, ys, color=vcols[v])
        yss[vcols[v]].append(ys)

    for i, col in enumerate(vcolsuniq):
        ax = axs[col]
        fig = figs[col]

        z = np.mean(yss[col], axis=0)
        m = len(yss[col])
        ax.plot(xs, z, color='k')

        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Shift')
        ax.set_ylabel(ylbl)
        plotpath = pjoin(outdir, '{}_comm{}.png'.format(lbl, i))
        # ax.text(.9, .9, '{} lines'.format(m),
                 # horizontalalignment='center', verticalalignment='center',
                 # transform = ax.transAxes)
        fig.savefig(plotpath, bbox_inches='tight'); plt.close()

##########################################################
def run_experiment(top, runid, coincexp, maxdist, outrootdir):
    info(inspect.stack()[0][3] + '()')

    info(top)
    outdir = pjoin(outrootdir, '{:03d}'.format(maxdist)) # Create output folders
    dirlayout1 = pjoin(outdir, 'netorig'); os.makedirs(dirlayout1, exist_ok=True)
    dirlayout2 = pjoin(outdir, 'netcoinc'); os.makedirs(dirlayout2, exist_ok=True)
    dirlayout3 = pjoin(outdir, 'netcoincgiant'); os.makedirs(dirlayout3, exist_ok=True)
    dirsignat = pjoin(outdir, 'signatures'); os.makedirs(dirsignat, exist_ok=True)

    random.seed(runid); np.random.seed(runid) # Random seeds

    isgraphml = top.endswith('.graphml')
    gid = os.path.basename(top).replace('.graphml', '') if isgraphml else top

    n = 2000; k = 6
    # n = 500; k = 6
    g, adj = generate_graph(top, n, k) # Create graph
    g.vs['origid'] = list(range(g.vcount()))
    info('n:{},k:{:.02f}'.format(g.vcount(), np.mean(g.degree())))

    if 'x' in g.vertex_attributes(): # Set the spatial layout
        aux = np.array([g.vs['x'], g.vs['y']]).T
        xy = [[x, -y] for x, y in aux]
    else:
        xy = np.array(g.layout(layout='fr'))
        g.vs['x'], g.vs['y']= xy[:, 0], xy[:, 1]

    # Coincidence between feature vectors
    pklpath = pjoin(outdir, '{}_{:02d}_coinc.pkl'.format(gid, runid))
    if isfile(pklpath):
        coinc0 = pickle.load(open(pklpath, 'rb'))
    else:
        vfeats, featlbls = extract_features(adj, g, isgraphml)
        coinc0 = get_coincidx_values(vfeats, coincexp, False)
        pickle.dump(coinc0, open(pklpath, 'wb'))

        # Plot deg distrib
        deg = vfeats[:, 0]
        d = np.diff(np.unique(deg)).min()
        left_of_first_bin = deg.min() - float(d)/2
        right_of_last_bin = deg.max() + float(d)/2
        plt.hist(deg, np.arange(left_of_first_bin, right_of_last_bin + d, d)); plt.xlabel('Degree')
        plt.savefig(pjoin(outdir, '{}_{:02d}_distribdeg.png'.format(gid, runid)))
        plt.close()

        # Plot nneigh distrib
        nneigh = vfeats[:, 0]
        plt.hist(nneigh, bins=20); plt.xlabel('Number of nearby nodes')
        plt.savefig(pjoin(outdir, '{}_{:02d}_distribnneigh.png'.format(gid, runid)))
        plt.close()

        # Plot coinc distrib
        plt.hist(coinc0.flatten()); plt.xlabel('Coincidence index')
        plt.savefig(pklpath.replace('.pkl', '.png')); plt.close()

    netorig = pjoin(outdir, '{}_{:02d}.pdf'.format(gid, runid))
    plotpath = pjoin(outdir, '{}_{:02d}_autorel.png'.format(gid, runid))
    lbl = '{}_{:02d}'.format(gid, runid)

    vsz = 7
    coords1 = plot_graph(g, xy, None, vsz, None, netorig)

    # Auto relation considering the coincidence betw the features
    means, stds = calculate_autorelation(g, coinc0, maxdist)
    plot_curves_and_avg(means, '', plotpath)

    # for coincthresh in np.arange(.5, .99, .02):
    # for coincthresh in [.78]: # TODO: REMOVE THIS
    for coincthresh in [.80, .85, .90, .95]: # TODO: REMOVE THIS
        expidstr = '{}_T{:.02f}_{:02d}'.format(gid, coincthresh, runid)
        info(expidstr)

        # Define plots paths
        netcoinc1 = pjoin(dirlayout1, '{}_netorig.png'.format(expidstr))
        netcoinc2 = pjoin(dirlayout2, '{}_netcoinc.png'.format(expidstr))
        netcoinc3 = pjoin(dirlayout3, '{}_netcoincgiant.png'.format(expidstr))
        if isfile(netcoinc1) and isfile(netcoinc2): continue

        coinc = get_coincidx_values(means, coincexp, False)
        coinc = threshold_values(coinc, coincthresh)
        gcoinc = igraph.Graph.Weighted_Adjacency(coinc, mode='undirected')
        gcoinc.vs['origid'] = g.vs['origid']

        plot_graph(gcoinc, None, None, g.vs.degree(), None, netcoinc2)
        giant = gcoinc.components().giant()
        lbls = [str(id) for id in giant.vs['origid']]

        origids = list(giant.vs['origid'])
        comm = giant.community_multilevel() # comm = giant.community_label_propagation()
        membs = comm.membership

        # Order membership id by cluster size
        _, counts = np.unique(membs, return_counts=True)
        idmap = {x: newid for newid, x in enumerate(reversed(np.argsort(counts)))}
        membs = np.vectorize(idmap.get)(membs)

        clrnotgiant = '#FFFFFF'
        clrsmallcomm = '#BBBBBB'

        vcols = []
        for i in range(g.vcount()):
            if not (i in origids): # Outside of the giant component
                vcol = clrnotgiant
            else:
                memb = membs[origids.index(i)]
                if memb >= len(PALETTE): # In a small community
                    vcol = clrsmallcomm
                else:
                    vcol = PALETTE[membs[origids.index(i)]]
            vcols.append(vcol)

        plot_graph(giant, None, None, giant.vs.degree(),
                   np.array(vcols)[giant.vs['origid']].tolist(), netcoinc3)

        plot_graph(g, coords1, None, vsz, vcols, netcoinc1)
        plot_curves_and_avg_comm(means, '', vcols, expidstr, dirsignat)
    # myutils.append_to_file('Counts of neighbor inside radius {}'.format(g.vs['featrad']),
                           # pjoin(outdir, 'README.md'))

##########################################################
def main(cfgpath, nprocs, readmepath, outrootdir):
    info(inspect.stack()[0][3] + '()')

    cfg = json.load(open(cfgpath))

    coincexp = cfg['coincexp']
    tops = cfg['tops']
    nruns = cfg['nruns'][0]
    maxdist = cfg['maxdist']
    runids = range(nruns)

    argsconcat = []
    for top in tops: # Can be either 'sbm' or a path to directory with graphml files
        if top == 'sbm':
            outdir = pjoin(outrootdir, 'sbm')
            args = list(product(['sbm'], runids, coincexp, maxdist, [outdir]))
            argsconcat.extend(args)
        else:
            top = top.replace('$HOME', os.environ['HOME'])
            for f in sorted(os.listdir(top)):
                if not f.endswith('.graphml'): continue
                fid = f.replace('.graphml', '')
                graphpath = pjoin(top, f)
                outdir = pjoin(outrootdir, fid)
                args = list(product([graphpath], runids, coincexp, maxdist, [outdir]))
                argsconcat.extend(args)

    parallelize(run_experiment, nprocs, argsconcat)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/sbm.json', help='Experiments settings')
    parser.add_argument('--nprocs', default=1, type=int, help='Number of procs')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)
    shutil.copy(args.config, args.outdir)
    main(args.config, args.nprocs, readmepath, args.outdir)
    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
