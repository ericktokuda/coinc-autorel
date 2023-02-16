import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for q in ['Theology', 'Physics', 'Fields_of_history']:
    df1 = pd.read_csv('./{}/2014/means.tsv'.format(q), sep='\t')
    df2 = pd.read_csv('./{}/2018/means.tsv'.format(q), sep='\t')

    dffilt = df1.merge(df2, how='inner', on='wid')
    # df2filt = df2.merge(df1, how='inner', on='wid')

    colsdata1 = [x+'_x' for x in df1.columns[2:].values]
    colsdata2 = [y+'_y' for y in df1.columns[2:].values]

    diff = dffilt[colsdata1].values - dffilt[colsdata2].values
    diff = np.abs(diff)
    wids = dffilt.wid.values

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    m = diff.shape[1]

    for i in range(diff.shape[0]):
        ax.plot(range(m), diff[i, :])

    outpath = './{}/diff.png'.format(q)
    plt.savefig(outpath)

    pd.DataFrame(dffilt).to_csv('./{}/wids.tsv'.format(q), index=False, columns=['wid'])
    pd.DataFrame(diff).to_csv('./{}/diff.tsv'.format(q), sep='\t', index=False, header=False)

    sums = np.sum(diff, axis=1)
    plt.close()
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.hist(sums, bins=50)
    outpath = './{}/hist.png'.format(q)
    plt.savefig(outpath)

    vspath = '/home/tokuda/temp/results/autorelation/20230206-coinc/2018/networks/{}_vs.tsv'.format(q)
    gpath = '/home/tokuda/temp/results/autorelation/20230206-coinc/2018/networks/{}_es.tsv'.format(q)
    dfvs = pd.read_csv(vspath, sep='\t')
    dfes = pd.read_csv(gpath, sep='\t')
    n = np.max(dfes.values) + 1

    import igraph

    g = igraph.Graph(n)
    g.add_edges(dfes.values)
    g.vs['wid'] = df1.wid
    g.simplify()

    g.vs['sz'] = 0.1
    wids2 = g.vs['wid']
    for i, wid in enumerate(dffilt.wid):
        # if not (wid in wids):
        idx = g.vs.select(wid=wid).indices[0]
        g.vs[idx]['sz'] = sums[i]
        # if not (wid in wids):
            # idx = g.vs.select(wid=wid).indices[0]
            # g.delete_vertices([idx])

    vszs = np.array(g.vs['sz']) * 10
    min0 = np.min(vszs)
    max0 = np.max(vszs)
    vszs = (vszs - min0) / (max0 - min0)
    vszs = vszs * 10 + 5
    vcols = 'blue'
    # print(np.min(vszs), np.max(vszs))
    # vszs = (vszs - min0) / (max0 - min0)
    # vszs = (vszs + 5

    igraph.plot(g, './{}/graph.png'.format(q), vertex_size=vszs, vertex_color=vcols)


# df3 = df1filt

