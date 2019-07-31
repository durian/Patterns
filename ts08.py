#
# -----------------------------------------------------------------------------
# Example:
#  python3 ts08.py -f 8752 -m84 -i -p 2011-01
#
# Plots claims in 24 warranty months for fgrp 8752, starting from 2011-01
#   using data from 3dprodmonths_2011-01_fgrp8752_84m.csv
#                             -p 2011-01  -f 8752 -m84
#   needs to be generated with plot_claims_proddate6.py -f 8752 -p 2011-01 -m84
#   if not present.
#
# Use the -c option to binarise values, and group
# -----------------------------------------------------------------------------
#
import re
import sys, os
import argparse
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
from collections import Counter

from read_data import *
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

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

# generate these with:
# python3 plot_claims_proddate4.py -f 3111 -p 2014-01 -m48
# python3 plot_claims_proddate4.py -f 2346 -p 2014-01 -m48
# python3 plot_claims_proddate4.py -f 2846 -p 2014-01 -m48
# python3 plot_claims_proddate4.py -f 2584 -p 2014-01 -m48
    
parser = argparse.ArgumentParser()
parser.add_argument( '-a', "--anon", action='store_true', default=False, help='Anomymise' )
parser.add_argument( '-c', "--chunks", type=int, default=None, help='Chunks' )
parser.add_argument( '-C', "--clusters", type=int, default=3, help='Clusters' )
parser.add_argument( '-f', '--fgrp', type=str, default="3111", help='Specify FGRP4s')
parser.add_argument( '-i', "--info", action='store_true', default=False, help='Info plot only' )
parser.add_argument( '-m', "--months", type=int, default=120, help='Plot this many months' ) # from datafile name
parser.add_argument( '-p', "--start_month", type=str, default="2010-01", help='Start month' )
args = parser.parse_args()

from shlex import quote
invocation = "python3 " + ' '.join(quote(s) for s in sys.argv)
with open( "invocation_ts08_log.txt", "a") as f:
    f.write( invocation + "\n" )

#ms = 120
#sd = "2010-01"
ms = args.months #120
sd = args.start_month #"2010-01"
fgrp_list = args.fgrp.split(",")
X = [] # np array
prod_months = []
#for fgrp in [3111, 3110, 2846, 2346, 2584, 8752]:
for fgrp in fgrp_list:
    try:
        fn = "3dprodmonths_{}_fgrp{}_{}m.csv".format( sd, fgrp, str(ms) )
        my_ts = pd.read_csv( fn, sep=";", index_col=0 )
        print( fn )
        prod_months = my_ts.index.values
    except FileNotFoundError:
        print( "Not found", fn )
        print( "-> python3 plot_claims_proddate6.py -f {} -p {} -m{}".format(fgrp, args.start_month, args.months) )
        continue
    # all 24 months of claims after each production month in data
    my_ts = my_ts.iloc[:, 0:25].divide(my_ts.iloc[:,-1], axis = 'rows').fillna(0) #normalise on warranty vol, x 1000
    X1 = my_ts.values[:,0:25]
    X1 = X1 * 1000.0
    try:
        X = np.concatenate((X, X1))
        print( X.shape )
    except ValueError:
        X = X1
        print( X.shape )

if len(X) == 0:
    sys.exit(1)
    
print( X.shape )
seed = 42
num = args.clusters
sz = X.shape[1]

def trend(y, x): # y comes after x in time
    # round x and y ?
    if y > x:
        return 1
    elif y < x:
        return -1
    return 0

def simplify(x):
    # filter
    if x > 1:
        return x
    return 0

def compact(values, dec=2):
        fmt = "{:4."+str(dec)+"f}"
        tmp = [ fmt.format(x) for x in values ]
        res =" ".join(tmp)
        return res
    
X_chunked = []
if args.chunks:
    for i,xx in enumerate(X):
        X1        = xx
        X1_len    = len( X1 )
        X1_chunks = X1_len // args.chunks
        print( "->", X1_len, X1_chunks )
        print( [ (x,x+args.chunks) for x in range( 0, X1_len, args.chunks ) ]  )
        X2        = [ sum(X1[x:x+args.chunks]) for x in range( 0, X1_len, args.chunks ) ] # sum chunks
        #X2        = [ sum(X1[x:x+args.chunks])/args.chunks for x in range( 0, X1_len, args.chunks ) ] # ave chunks
        print( X1 )
        print( X2 )
        # Bin/trend them as well?
        X2_trend = [0] + [ trend(X2[n],X2[n-1]) for n in range(1, len(X2)) ] # compare to previous
        print( X2_trend )
        X2_simple = [ simplify(x) for x in X2 ] 
        print( X2_simple)
        X2 = X2_trend #X2_simple #X2_trend
        X_chunked.append( X2 )
    X = np.asarray(X_chunked)
    
# ----

# Does a boxplot make sense?
# Every month we plot has 24 (okay 25)  months of claims, so each box depicts 25
# data points. It shows "all the first months after production, all second months after
# production", etc.
green_diamond = dict(markerfacecolor='g', marker='D', markersize=3)
medianprops   = dict(linewidth=0)
meanlineprops = dict(linestyle='-', linewidth=1, color='purple')
fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
ax2.set_title( "Claims after $x$ months after production month" )
boxes = ax2.boxplot(X, flierprops=green_diamond,
                    medianprops=medianprops,
                    #showfliers=False,
                    meanprops=meanlineprops, meanline=True, showmeans=True)
outliers = [flier.get_ydata() for flier in boxes["fliers"]]
print( "Boxplot outliers" )
for i, outlier in enumerate(outliers):
    print( "{:2n} {:2n} {}".format(i, len(outlier), compact(outlier)) )
if args.anon:
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    ax2.set_yticklabels(empty_string_labels)
    labels = [item.get_text() for item in ax2.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax2.set_xticklabels(empty_string_labels)
#top_points = boxes["fliers"][0].get_data()[1]
#print( top_points )
if args.anon:
    pngfile = "ts8boxplot_fgrp"+str(args.fgrp)+"_"+args.start_month+"_"+str(args.months)+"m_anon.png"
else:
    pngfile = "ts8boxplot_fgrp"+str(args.fgrp)+"_"+args.start_month+"_"+str(args.months)+"m.png"
if os.path.exists( pngfile ):
    os.remove( pngfile )
fig2.savefig(pngfile, dpi=300)
print( "Saved", pngfile )

# ----

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
lines = []
fgrps = []
topn  = Counter()
date_labels = [ x for x in range( 0, X.shape[1] ) ]
title_str = "Claims after $x$ months after production month"
print( date_labels )
import matplotlib.patches as mpatches # for legend
patchList = [] # for legend
patch_years = []
for i,xx in enumerate(X): # this hould be in fgrp order
    #                  (i//12) to get some colour per year
    c = cat20_colours[ (i//12) % len(cat20_colours) ] #"-k" #cat20_colours[ i % len(cat20_colours) ]
    if not (prod_months[i])[:-6] in patch_years: # -6 to remove day and month to keep year
        data_key = mpatches.Patch( color=c, label=(prod_months[i])[:-6] )
        patchList.append(data_key)
        patch_years.append( (prod_months[i])[:-6] )
    jumps = [ abs(xx[n]-xx[n-1]) for n in range(1, len(xx)) ] # diffs
    topn[ prod_months[i] ] = max(jumps) # save highest
    #print( prod_months[i], max(jumps) )
    jumps = sum( [1 if abs(j)>=1 else 0 for j in jumps] ) # counts jumps >= jump_size
    #print( jumps )
    if jumps > 999: # in the claim date we might want 3 or so
        line, = ax.plot( date_labels, xx, c, alpha=.8, marker='.' ) # plot thicker/marker
        lines.append( line )
    else:
        ax.plot( date_labels, xx, c, alpha=.5 )
ax.set_title( title_str )
if not args.anon:
    ax.legend(handles=patchList)
#
#ax.xaxis.set_minor_locator( MonthLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]) )
#ax.xaxis.set_major_locator( YearLocator() )
#ax.xaxis.set_major_formatter( DateFormatter('%Y-%m') )
#fig.autofmt_xdate()
plt.tight_layout()

if args.anon:
    labels = [item.get_text() for item in ax.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_yticklabels(empty_string_labels)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_xticklabels(empty_string_labels)
if args.anon:
    pngfile = "ts8plot_fgrp"+str(args.fgrp)+"_"+args.start_month+"_"+str(args.months)+"m_anon.png"
else:
    pngfile = "ts8plot_fgrp"+str(args.fgrp)+"_"+args.start_month+"_"+str(args.months)+"m.png"
if os.path.exists( pngfile ):
    os.remove( pngfile )
fig.savefig(pngfile, dpi=300)
print( "Saved", pngfile )

if args.info:
    plt.show(block=True)
    sys.exit(1)

# ----

km = TimeSeriesKMeans(n_clusters=num, verbose=True, random_state=seed)
y_pred = km.fit_predict(X)
print( y_pred )

plt.figure( figsize=(8, 2*num) )
for yi in range(num):
    plt.subplot(num, 1, yi + 1)
    for xx in X[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    #plt.xlim(0, sz)
    #plt.ylim(0, 4)
    if yi == 0:
        plt.title("Euclidean $k$-means")
plt.tight_layout()

# ----

print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=num, n_init=2, metric="dtw", verbose=True,
                          max_iter_barycenter=10, random_state=seed)
y_pred = dba_km.fit_predict(X)
print( y_pred )

plt.figure()
for yi in range(num):
    plt.subplot(num, 1, 1 + yi)
    for xx in X[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    #plt.ylim(0, 4)
    if yi == 0:
        plt.title("DBA $k$-means")
plt.tight_layout()

# ----

print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=num, metric="softdtw", metric_params={"gamma_sdtw": .01},
                           verbose=True, random_state=seed)
y_pred = sdtw_km.fit_predict(X)
print( y_pred )

plt.figure()
for yi in range(num):
    plt.subplot(num, 1, 1 + yi)
    for xx in X[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    #plt.ylim(0, 4)
    if yi == 0:
        plt.title("Soft-DTW $k$-means")
plt.tight_layout()

plt.show()
