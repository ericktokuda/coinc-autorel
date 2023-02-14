#!/usr/bin/env python3
"""one-line docstring
"""

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import pandas as pd

##########################################################
def plot_results(df, outdir):
    info(inspect.stack()[0][3] + '()')
    plotdir = pjoin(outdir, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    feats = ['ncomps', 'szmax', 'szmean', 'szstd', 'degmeanmean', 'degmeanstd',
             'degstdmean', 'degstdstd', 'mplmean', 'mplstd',
             'transmean', 'transstd']

    models = ['er', 'gr', 'ba', 'sb']
    # ns = [200, 350, 500]
    ns = [350]
    ks = [6, 12, 18]
    for n in ns:
        for k in ks:
            for feat in feats:
                plotpath = pjoin(plotdir, 'n{}_k{}_{}.png'.format(n, k, feat))

                plt.close()
                fig, ax = plt.subplots()
                skip = False
                for model in models:
                    z = df.loc[(df.nreq == n) & (df.k == k) & (df.model == model)][feat]

                    if len(z) == 0:
                        skip = True
                        continue

                    aux = df.loc[df.model == model][feat]
                    aux.plot.kde(ax=ax, label=model, legend=True)

                if skip: continue

                ax.set_ylabel('')
                ax.set_xlabel(feat[0].upper() + feat[1:])
                plt.savefig(plotpath)

##########################################################
def main(dfpath, outdir):
    df = pd.read_csv(dfpath)
    plot_results(df, outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--res', required=True, help='Results (csv) path')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)
    main(args.res, args.outdir)
    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
