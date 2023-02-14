#!/usr/bin/env python3
"""Utility functions
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
def get_common_ids(nodespath1, nodespath2, outdir):
    """Get common ids between the two list of nodes"""
    info(inspect.stack()[0][3] + '()')
    f = os.path.basename(nodespath1)
    outpath = pjoin(outdir, f)

    df1 = pd.read_csv(nodespath1, sep='\t')
    df2 = pd.read_csv(nodespath2, sep='\t')
    df3 = df1.merge(df2, how='inner', on='wid')
    df3['title'] = df3.title_x
    df3.to_csv(outpath, index=False, sep='\t', columns=['wid', 'title'])

##########################################################
def main(nodespath1, nodespath2, outdir):
    info(inspect.stack()[0][3] + '()')
    get_common_ids(nodespath1, nodespath2, outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes1', required=True, help='Nodes list 1')
    parser.add_argument('--nodes2', required=True, help='Nodes list 2')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.nodes1, args.nodes2, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
