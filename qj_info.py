#
# -----------------------------------------------------------------------------
# Example:
#  python3 qj_info.py -f allqjs_b3g1a3_mnl1_mxl1096_2010-01_fgrp2_120m_prod.csv
#
# produced with: python3 ts06hitrates.py -f 2 -m 120 -p 2010-01 -d prod (-g9)
#
# plots with:
#  python3 ts06hitrates.py -f 2 -m 120 -p 2010-01 -d prod --min_qj_len 1 --max_qj_len 15 --xmin 0 --xmax 15
#
# -----------------------------------------------------------------------------
#
import re
import sys, os, pickle
import argparse
import glob
from datetime import datetime
from datetime import timedelta
#https://stackoverflow.com/questions/35066588/is-there-a-simple-way-to-increment-a-datetime-object-one-month-in-python
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib as mpl
mpl.use("Qt5Agg") #TkAgg crashes
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
from read_data import *
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf
from pandas.plotting import lag_plot
import itertools
from collections import Counter
from matplotlib.dates import MonthLocator, YearLocator, DateFormatter

from scipy import stats
import calendar

cat20_colours = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", 
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    #https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    #"#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    #"#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
    "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"                 
]

parser = argparse.ArgumentParser()
parser.add_argument( '-a', "--anon", action='store_true', default=False, help='Anonymous axes' )
parser.add_argument( '-f', '--fn', type=str, default=None, help='File')
parser.add_argument( '-l', '--lengths', type=str, default="14,14,30,30,90,180,180,360", help='Lengt of QJs')
args = parser.parse_args()

#filenames = glob.glob( args.fn )
#print( filenames )

#from shlex import quote
#invocation = "python3 " + ' '.join(quote(s) for s in sys.argv)
#with open( "invocation_qj_info_log.txt", "a") as f:
#    f.write( invocation + "\n" )

qjs_df = pd.read_csv( args.fn, sep=";", index_col=0 )
print( qjs_df.head(1) )
good_count = qjs_df['hitrate'][qjs_df['hitrate'] < 1.0 ].count()
bad_count  = qjs_df['hitrate'][qjs_df['hitrate'] >= 1.0 ].count()
marker = ""
if good_count > bad_count:
    marker = "!"
print( good_count, bad_count, marker )

# Length info
lengths = [ int(x) for x in args.lengths.split(",") ]
min_l = 1
max_l = min_l
       #   1 -  15   91:  50  41 !
print( "   <= d <   num    +   -" )
for length in lengths:
    max_l += length
    sub_df = qjs_df[ (qjs_df["days"] >= min_l) &  (qjs_df["days"] < max_l) ]
    good_count = sub_df['hitrate'][sub_df['hitrate'] < 1.0 ].count()
    bad_count  = sub_df['hitrate'][sub_df['hitrate'] >= 1.0 ].count()
    marker = ""
    if good_count > bad_count:
        marker = "!"
    print( "{:4n} -{:4n} {:4n}: {:3n} {:3n} {:.4f} {}".format(
        min_l, max_l, sub_df.shape[0], good_count, bad_count, good_count/bad_count, marker)
    )
    min_l = max_l
