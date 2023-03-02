#!/usr/bin/env python3
"""Filter snapshot by a list of ids. Expected input is in consonni's format.
Output is the format of the output of categories.py. The input format is just
the list of edges so we filter the edges. As such, the nodes of the output have
at least one connection.
Output a list of vertices and edges because we may have more than one connected
component.
"""

import argparse
import time, datetime
import os, sys, inspect, pickle
from os.path import join as pjoin
from os.path import isfile
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import pandas as pd

##########################################################
def load_dataframe(dfpath):
    info(inspect.stack()[0][3] + '()')
    sep = '\t'
    pklpath = dfpath.replace('.csv', '.pkl').replace('.tsv', '.pkl')
    if os.path.isfile(pklpath):
        return pickle.load(open(pklpath, 'rb'))
    elif os.path.isfile(dfpath):
        df = pd.read_csv(dfpath, sep=sep, low_memory=False)
        pickle.dump(df, open(pklpath, 'wb'))
        return df

##########################################################
def get_out_page_ids(queryid, df):
    return df.loc[df.page_id_from == queryid]['page_id_to'].unique()

##########################################################
def filter_df_by_ids(df, ids, removeid=-1):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    dffilt = df.loc[df.isin(ids).page_id_from & df.isin(ids).page_id_to]
    dffilt = dffilt[dffilt.page_id_from != removeid]
    dffilt = dffilt[dffilt.page_id_to != removeid]
    return dffilt

##########################################################
def get_id_all_pages(df):
    return df[['page_id_from', 'page_id_to']].values.flatten()

##########################################################
def main(snappath, idspath, outdir):
    info(inspect.stack()[0][3] + '()')

    info('snappath:{}'.format(snappath))
    info('idspath:{}'.format(idspath))

    f = os.path.basename(idspath)
    if '_max2.tsv' in f:
        suff = '_'.join(f.split('_')[:-1])
    else:
        suff = f.split('.')[0]

    outpath1 = pjoin(outdir, '{}_vs.tsv'.format(suff))
    outpath2 = pjoin(outdir, '{}_es.tsv'.format(suff))

    if isfile(outpath2):
        return

    if type(snappath) == str: # Load snapshot
        if isfile(snappath.replace('.tsv', '.pkl')):
            dfsnap = pickle.load(open(snappath.replace('.tsv', '.pkl'), 'rb'))
        else:
            dfsnap = pd.read_csv(snappath, sep='\t', low_memory=False)
    else:
        dfsnap = snappath

    if type(idspath) == str: # Load ids
        dfids = pd.read_csv(idspath, sep='\t', low_memory=False)
    else:
        dfids = idspath

    if 'ns' in dfids.columns: # Filter by namespace 0 (pages)
        dfids = dfids.loc[dfids.ns == 0]

    # Filter edges by list of ids
    cls1 = dfsnap.page_id_from.isin(dfids.pageid)
    cls2 = dfsnap.page_id_to.isin(dfids.pageid)
    filt = dfsnap.loc[cls1 & cls2]

    info('{} edges'.format(len(filt)))

    aux = filt[['page_id_from', 'page_id_to']].values.flatten()
    wids = np.unique(sorted(aux))
    dfwids = pd.DataFrame(wids, columns=['wid'])
    info('{} vertices'.format(len(wids)))

    wid2vid = {wid:i for i, wid in enumerate(wids)}

    rows1 = filt[['page_id_from', 'page_title_from']].values
    rows2 = filt[['page_id_to', 'page_title_to']].values
    dfpages = pd.DataFrame(np.vstack((rows1, rows2)), columns=['wid', 'title'])
    dfpages = dfpages.sort_values(['wid']).drop_duplicates()
    dfpages.to_csv(outpath1, sep='\t', index=False)

    c1 = filt.page_id_from.map(wid2vid)
    c2 = filt.page_id_to.map(wid2vid)

    filt = pd.DataFrame([c1, c2]).transpose()
    filt.to_csv(outpath2, sep='\t', header=['src', 'tgt'], index=False)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snappath', required=True, help='Path to the Consoni\'s snapshot')
    parser.add_argument('--idspath', required=True, help='Path to the ids to be used to filter the snapshot. Expected field "pageid"')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.snappath, args.idspath, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
