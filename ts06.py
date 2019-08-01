#
# -----------------------------------------------------------------------------
# Example:
#  python3 ts06.py -m 120 -p 2010-01 --dl 2014-01 --dh 2019-01 -d prod -f 5 -i
#
# Plots claims per month for fgrps in 5... between 2014-01 and 2019-01,
#   using data from "allclaimsprodmonth_2010-01_fgrp5111_120m.csv"
#                          -d prod   -p 2010-01  -f 5  -m120
#   for all the fgrps in the argument
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
from read_data import *
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf
from pandas.plotting import lag_plot
import itertools
from collections import Counter
from matplotlib.dates import MonthLocator, YearLocator, DateFormatter

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape

from scipy.signal import find_peaks, find_peaks_cwt
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
parser.add_argument( '-b', "--block", action='store_true', default=False, help='Show plots' )
parser.add_argument( "--adf", action='store_true', default=False, help='Print ADF stats' )
parser.add_argument( "--autocorr_plot", action='store_true', default=False, help='Plot autocorrelation plots' )
parser.add_argument( '-f', '--fgrp', type=str, default=3, help='Specify FGRP4s')
parser.add_argument( '-F', '--notfgrp', type=str, default="", help='FGRP4s to ignore')
parser.add_argument( '-d', '--date_type', type=str, default="claim", help='Date type, "prod" or "claim" (default)')
parser.add_argument( '-c', "--chunks", type=int, default=0, help='Chunks' )
parser.add_argument( '-C', "--clusters", type=int, default=3, help='Clusters' )
parser.add_argument( '-i', "--info", action='store_true', default=False, help='Info plot only' )
parser.add_argument( '-j', "--jump_size", type=float, default=2, help='Count jump if diff is larger' )
parser.add_argument( '-J', "--jump_count", type=int, default=1, help='Add to legend/peaks if more jumps than this' )
parser.add_argument( '-k', "--kmeans_algo", type=int, default=0, help='k-Means algorithm' )
parser.add_argument( '-l', "--legend", type=int, default=10, help='Plot legend if less than this (10)' )
parser.add_argument( '-m', "--months", type=int, default=48, help='Plot this many months' ) # from datafile name
parser.add_argument( '-M', "--margin", type=float, default=0.1, help='Margin for chunkifier (0.1)' )
parser.add_argument( "--name", type=str, default=None, help='Name instead if fgrps in filename' ) 
parser.add_argument( '-p', "--start_month", type=str, default="2012-01", help='Start month' )
parser.add_argument( '-P', "--peaks", action='store_true', default=False, help='Show peaks' )
parser.add_argument( '-y', "--ymax", type=float, default=None, help='Y-max for axis' )
parser.add_argument( "--dl", type=str, default="2016-01", help='Date lo' )
parser.add_argument( "--dh", type=str, default="2018-01", help='Date hi' )
parser.add_argument( '-v', "--verbose", action='store_true', default=False, help='Verbose' )
args = parser.parse_args()

from shlex import quote
invocation = "python3 " + ' '.join(quote(s) for s in sys.argv)
with open( "invocation_ts06_log.txt", "a") as f:
    f.write( invocation + "\n" )

# generate these with:
# python3 plot_claims_proddate4.py -f 3111 -p 2014-01 -m48
# python3 plot_claims_proddate4.py -f 2346 -p 2014-01 -m48
# python3 plot_claims_proddate4.py -f 2846 -p 2014-01 -m48
# python3 plot_claims_proddate4.py -f 2584 -p 2014-01 -m48
# for x in 2584 3110 3111 2346 2581 2584 2841 2846 3902 3651 2841 5931 8752; do python3 plot_claims_proddate4.py -f $x -p 2014-01 -m48;done
# for x in 2584 3110 3111 2346 2581 2584 2841 2846 3902 3651 2841 5931 8752; do python3 plot_claims_proddate4.py -f $x -p 2011-01 -m108;done

def delta(x):
    if x > args.margin: # should compensate for chunks?
        return args.margin
    elif x < -args.margin:
        return -args.margin
    return 0

def delta0(x):
    if x > 2*args.margin: # should compensate for chunks?
        return 2
    elif x > args.margin:
        return 1
    elif x < -2*args.margin:
        return -2
    elif x < -args.margin:
        return -1
    return 0

ms = args.months
sd = args.start_month
#
X = np.asarray([]) # np array
chunks = args.chunks

if args.name:
    fname_fgrp = args.name
else:
    fname_fgrp = args.fgrp # for filename
    if len(fname_fgrp) > 200:
        fname_fgrp = "_LONG_"
    
# Better have between two boundaries? or match on 3, or 31, etc in beginning
try:
    if int(args.fgrp) < 10:
        if int(args.fgrp) == 1:
            args.fgrp = "1014,1113,1115,1122,1124,1126,1129,1142,1143,1144,1149,1152,1157,1162,1163,1167,1223,1413,1414,1415,1416,1512,1514,1515,1519,1613,1614,1615,1619,1632,1792,1832,1841,1871,1891,1914,1982"
        elif int(args.fgrp) == 2:
            args.fgrp = "2102,2105,2108,2111,2112,2115,2116,2117,2121,2125,2126,2129,2131,2132,2133,2134,2139,2141,2142,2143,2144,2145,2146,2148,2149,2151,2152,2153,2154,2155,2158,2161,2162,2163,2164,2165,2166,2167,2168,2169,2171,2172,2181,2184,2211,2212,2213,2214,2215,2219,2221,2222,2228,2229,2231,2303,2319,2331,2334,2337,2341,2342,2346,2348,2349,2361,2366,2371,2372,2373,2374,2375,2379,2384,2389,2431,2439,2441,2446,2459,2512,2513,2515,2516,2521,2522,2523,2524,2525,2527,2528,2529,2532,2533,2535,2538,2539,2542,2543,2545,2547,2548,2549,2551,2552,2561,2562,2563,2564,2565,2571,2572,2579,2581,2582,2584,2585,2586,2587,2588,2589,2611,2612,2614,2615,2616,2617,2618,2619,2621,2626,2627,2628,2629,2631,2632,2633,2634,2636,2637,2639,2651,2652,2663,2691,2711,2715,2731,2732,2741,2811,2841,2842,2846,2849,2861,2862,2869,2931,2932,2933,2934,2935,2939,2991"
        elif int(args.fgrp) == 3:
            args.fgrp = "3001,3002,3011,3019,3110,3111,3112,3113,3119,3121,3131,3139,3211,3212,3224,3241,3311,3318,3331,3334,3339,3341,3349,3513,3514,3519,3521,3522,3525,3526,3527,3531,3535,3538,3541,3552,3561,3565,3569,3571,3621,3622,3624,3629,3631,3634,3635,3637,3638,3639,3640,3641,3643,3644,3645,3646,3648,3649,3651,3662,3665,3669,3681,3689,3711,3712,3713,3714,3716,3719,3721,3722,3723,3725,3729,3731,3733,3734,3735,3739,3741,3742,3745,3749,3751,3752,3753,3754,3757,3758,3759,3761,3765,3769,3811,3819,3822,3829,3835,3843,3852,3861,3862,3864,3865,3869,3872,3902,3903,3911,3929,3931,3949,3952,3953,3971,3972,3982,3989"
        elif int(args.fgrp) == 4:
            args.fgrp = "4111,4112,4113,4114,4117,4122,4133,4134,4135,4136,4137,4138,4139,4144,4149,4212,4213,4219,4249,4311,4312,4313,4314,4315,4316,4317,4319,4321,4322,4323,4324,4325,4326,4327,4328,4329,4341,4343,4371,4372,4376,4378,4379,4384,4453,4511,4513,4514,4515,4531,4532,4533,4601,4602,4609,4611,4619,4651,4653,4654,4655,4656,4657,4658,4659,4661,4663,4664,4665,4666,4669,4671,4684,4811,4814,4816,4821,4824,4825,4832,4839,4911,4912,4953"
        elif int(args.fgrp) == 5:
            args.fgrp = "5111,5112,5113,5114,5115,5116,5117,5119,5121,5123,5124,5126,5129,5131,5134,5141,5211,5213,5222,5241,5514,5516,5611,5612,5614,5617,5618,5619,5621,5622,5629,5631,5633,5635,5639,5651,5653,5654,5655,5659,5711,5731,5739,5911,5921,5929,5931,5939"
        elif int(args.fgrp) == 6:
            args.fgrp = "6102,6112,6113,6119,6121,6122,6125,6126,6129,6411,6412,6413,6414,6419,6421,6422,6424,6428,6429,6431,6434,6436,6438,6439,6451,6452,6453,6454,6455,6456,6457,6459,6511,6521,6522,6523,6525,6527,6551,6553,6554,6555,6559,6562,6571,6999"
        elif int(args.fgrp) == 7:
            args.fgrp = "7111,7112,7113,7114,7116,7118,7119,7121,7123,7131,7149,7171,7173,7211,7212,7213,7214,7219,7221,7222,7229,7242,7252,7261,7262,7269,7271,7273,7281,7422,7611,7612,7613,7614,7621,7622,7629,7641,7644,7645,7647,7648,7661,7711,7712,7717,7721,7722,7731,7732,7735,7736,7739,7741,7761,7762,7999"
        elif int(args.fgrp) == 8:
            args.fgrp = "8013,8101,8102,8109,8111,8112,8114,8117,8121,8126,8131,8136,8154,8172,8181,8182,8189,8211,8212,8213,8219,8231,8232,8241,8251,8254,8259,8271,8311,8312,8315,8318,8321,8341,8342,8343,8344,8345,8349,8351,8352,8361,8365,8412,8415,8417,8431,8432,8433,8434,8435,8441,8444,8445,8451,8454,8457,8461,8462,8463,8469,8481,8511,8521,8525,8526,8529,8552,8554,8556,8561,8562,8579,8611,8615,8631,8639,8655,8659,8712,8715,8721,8724,8731,8732,8733,8734,8739,8741,8742,8743,8744,8746,8747,8748,8752,8761,8771,8781,8811,8812,8821,8825,8841,8845,8847,8851,8912,8913,8915,8916,8917,8918,8919,8921,8961,8962,8966,8969,8971,8979,8995,8999"
        elif int(args.fgrp) == 9:
            args.fgrp = "9131,9218,9219,9221,9222,9224,9225,9227,9229,9819,9862,9939,9999"
        else:
            args.fgrp = ",".join( [str(x) for x in range(1000,10000)] )
except ValueError:
    pass

fgrp_list    = args.fgrp.split(",")
notfgrp_list = args.notfgrp.split(",")
y_min =  100
y_max = -100
date_lo = args.dl
date_hi = args.dh
if args.anon:
    title_str = "Claims on "+args.date_type+" date"
else:
    title_str = "Claims on "+args.date_type+" date ("+date_lo+" -- "+date_hi+")"
param_str = "j_size "+str(args.jump_size)+", j_count "+str(args.jump_count)
if chunks > 0:
    param_str += ", c"+str(chunks)+", M"+str(args.margin)
    
date_lo_dt   = datetime.strptime(date_lo, '%Y-%m')
date_hi_dt   = datetime.strptime(date_hi, '%Y-%m')
r            = relativedelta( date_hi_dt, date_lo_dt )
delta_months = r.months + (r.years*12)
date_labels  = []

if chunks == 0:
    for m in range( 0, delta_months ): 
        prod_dt = date_lo_dt + relativedelta(months = m)
        #prod_date     = str(prod_dt)[0:7] #YYYY-MM
        date_labels.append( prod_dt )
else:
    for m in range( 0, delta_months // chunks ):
        prod_dt = date_lo_dt + relativedelta(months = m*chunks)
        #prod_date     = str(prod_dt)[0:7] #YYYY-MM
        date_labels.append( prod_dt )
        print( str(prod_dt)[0:7] )

read_fgrps = []
boxplotdata   = []
boxplotlabels = []
for fgrp in fgrp_list:
    if fgrp in notfgrp_list:
        continue
    try:
        if args.date_type == "claim":
            fn = "allclaimsclaimmonth_{}_fgrp{}_{}m.csv".format( sd, str(fgrp), str(ms) )
            my_ts = pd.read_csv( fn, sep=";", index_col=0 )
            my_ts = my_ts["normalised"] #[0:24]
        else:
            fn = "allclaimsprodmonth_{}_fgrp{}_{}m.csv".format( sd, str(fgrp), str(ms) )
            my_ts = pd.read_csv( fn, sep=";", index_col=0 )
            my_ts = my_ts["0"] #[0:24]
        my_ts = my_ts[ (my_ts.index >= date_lo) & (my_ts.index < date_hi) ]
        my_ts = my_ts.fillna(0)
        #print( my_ts )
        non_zero = my_ts.astype(bool).sum(axis=0)
        if non_zero < 1:
            if args.verbose:
                print( "SKIP", fn )
            continue
        if args.verbose:
            print( fn )
            print( fgrp, my_ts.max() )
        read_fgrps.append( fgrp ) #the ones which were ok
    except FileNotFoundError:
        if args.verbose:
            print( "Not found", fn )
        continue
    #
    boxplotdata.append( my_ts.values )
    boxplotlabels.append( fgrp )
    #
    X1 = my_ts.values.reshape(1,-1) #make it into one row
    X1 = np.nan_to_num(X1)
    #
    # Process ?
    if chunks > 0:
        X2        = X1[0]
        X2_len    = len( X2 )
        X2_chunks = X2_len // chunks
        X3        = [ sum(X2[x:x+chunks]) for x in range( 0, X2_len, chunks ) ] # sum chunks
        #X3        = [ sum(X2[x*chunks:(x+1)*chunks]) for x in range( 0, X2_chunks, chunks ) ] # sum chunks
        #X3        = [ sum(X2[x*chunks:(x+1)*chunks])/chunks for x in range( 0, X2_chunks, chunks ) ] # ave chunks
        print( X3 )
        X2 = X3
        #X2_diff = [ X2[n] - X2[n-1] for n in range(1, len(X2)) ]
        X2_diff = [0] + [ delta(X2[n]-X2[n-1]) for n in range(1, len(X2)) ] # compare to previous
        X2_diff = list( itertools.accumulate(X2_diff) ) # make running sum of -1,0,1 values
        print( X2_diff )
        y_min = min( min(X2_diff), y_min ) # for axis
        y_max = max( max(X2_diff), y_max )
        X1 = np.asarray( [X2_diff] )
    try:
        X = np.concatenate((X, X1))
    except ValueError: # first time when X is empty
        X = X1
        
print( X.shape )
#print( X )
seed = 42
num = args.clusters
sz = X.shape[1]-1

# Calculate average/stdev for auto jump size?
jump_precalc = np.array(0)
if args.jump_size == 0:
    #args.jump_size = boxplotdata.mean()
    jump_precalc = np.std(boxplotdata, axis=1) 
    
outlierprops = dict(marker='o', markerfacecolor='red', markersize=4, linestyle='none')
plot_len = max(len(boxplotdata)//6, 8)
plot_wth = 4 #max(plot_len // 2, 4)
fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(plot_len, plot_wth))
ax0.boxplot( boxplotdata, vert=True,
             meanline=True, showmeans=True,
             labels=boxplotlabels,
             flierprops=outlierprops)
for label in ax0.get_xticklabels():
    label.set_rotation(-80)
    label.set_fontsize(8)

if args.anon:
    labels = [item.get_text() for item in ax0.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    ax0.set_yticklabels(empty_string_labels)
    labels = [item.get_text() for item in ax0.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax0.set_xticklabels(empty_string_labels)

if args.anon:
    ax0.set_title( "Number of claims per function group" )
else:
    ax0.set_title( "Number of claims per function group ("+date_lo+" -- "+date_hi+")" )
ax0.set_xlabel( "Function group" )
ax0.set_ylabel( "Number of claims" )
plt.tight_layout()

pngfile = "ts6boxplot_"+args.date_type+"_"+date_lo+"-"+date_hi+"_fgrp"+str(fname_fgrp)+"_p"+args.start_month+"_m"+str(args.months)+"_"+args.date_type+".png"
if os.path.exists( pngfile ):
    os.remove( pngfile )
fig0.savefig(pngfile, dpi=300)
print( "Saved", pngfile )

print( "-" * 40 )

# ----

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
lines = []
fgrps = []
topn  = Counter()
peaks_hi   = Counter()
peaks_cwt  = []
adf_stats  = []
for i,xx in enumerate(X): # this should be in fgrp order
    c     = cat20_colours[ i % len(cat20_colours) ]
    if jump_precalc.any():
        args.jump_size = jump_precalc[i]*3 # not really working...
        print( "args.jump_size", args.jump_size )
    jumps = [ abs(xx[n]-xx[n-1]) for n in range(1, len(xx)) ] # diffs
    topn[read_fgrps[i]] = max(jumps) # save highest
    if args.verbose:
        print( read_fgrps[i], "max(jumps) =", max(jumps) )
    jumps = sum( [1 if abs(j)>=args.jump_size else 0 for j in jumps] ) # counts jumps >= jump_size
    #print( jumps )
    #
    # Data in xx
    #plt.hist(xx, bins="auto")
    #plt.show(block=True)
    #
    if args.autocorr_plot and i < 3:
        fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
        if args.anon:
            autocorrelation_plot(xx, ax=ax0)
            labels = [item.get_text() for item in ax0.get_yticklabels()]
            empty_string_labels = ['']*len(labels)
            ax0.set_yticklabels(empty_string_labels)
            labels = [item.get_text() for item in ax0.get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            ax0.set_xticklabels(empty_string_labels)
        else:
            autocorrelation_plot(xx, ax=ax0, label=read_fgrps[i])
        ax0.set_ylim(-1,1)
        ax0.set_ylim(-1,1)
        pngfile = "ts6plot_autocorr_"+args.date_type+"_"+date_lo+"-"+date_hi+"_fgrp"+str(fname_fgrp)+"_p"+args.start_month+"_m"+str(args.months)+"_"+args.date_type+".png"
        if os.path.exists( pngfile ):
            os.remove( pngfile )
        fig0.savefig(pngfile, dpi=300)
        print( "Saved", pngfile )
    #
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(xx, autolag='AIC') #The null hypothesis of the test is that the time series is non-stationary.
    if args.verbose:
        print( result )
    #print('ADF Statistic: %f' % result[0])
    #print('p-value: %f' % result[1])
    stationary = False
    if result[1] < 0.01:
        #print( "Stationary" )
        stationary = True
    else:
        pass
        #print( "Not stationary" )
    # also stationary if ADFStat > 1% (for 1% significance))
    #
    #print('samples: %f' % result[3])
    #print('Critical Values:')
    #for key, value in result[4].items():
    #    print('\t%s: %.3f' % (key, value))
    if args.verbose:
        print( [read_fgrps[i], result[0], result[1], result[2], result[3], stationary, result[4]["1%"]] )
    adf_stats.append( [read_fgrps[i], result[0], result[1], result[2], result[3], stationary, result[4]["1%"]] )
    '''
    diff = list()
    for i in range(1, len(xx)):
        value = xx[i] - xx[i - 1]
        diff.append(value)
    # line plot
    autocorrelation_plot(diff)
    plt.show()
    '''
    #
    if args.autocorr_plot:
        autocorr, confint, qstat, pvals = acf(xx, nlags=48, qstat=True, alpha=0.01)
        print( autocorr, confint, qstat, pvals )
    #
    for w in range(3,4):
        peakind = find_peaks_cwt(xx, [w])#np.arange(3,5)) # xx, widths
        if args.verbose:
            print( "find_peaks_cwt", w, peakind )
        #for p in peakind:
        #    ax.vlines(date_labels[p], ymin=0, ymax=xx[p], color=cat20_colours[ w % len(cat20_colours) ])
    #
    peaks, props = find_peaks(xx, height=args.jump_size, distance=2, width=1)
    if len(peaks) >= args.jump_count:
        if args.verbose:
            print( "find_peaks", args.jump_size, peaks, props ) #props["peak_heights"], props["prominences"] )
        x_labels = []
        for p in peaks:
            x_labels.append( date_lo_dt + relativedelta(months = p) )
        if args.verbose:
            print( "->", peaks, [str(x)[0:10] for x in x_labels], xx[peaks] - props["prominences"], xx[peaks] )
        peaks_hi[read_fgrps[i]] = len(peaks)
        #
        # This plots an x at the peaks. also saves as legend (instead of the original jumps legend)
        if args.peaks:
            line, = ax.plot(x_labels, xx[peaks], "x", color=c)
            lines.append( line )
            fgrps.append( read_fgrps[i] )  #these appear in the legend as well, and are printed below
        if False:
            ax.vlines(x=x_labels, ymin=xx[peaks] - props["prominences"], ymax = xx[peaks], color = c)
        #
        xmin_labels = []
        xmax_labels = []
        for p in props["left_ips"]:
            xmin_labels.append( date_lo_dt + relativedelta(days=p*30.5) ) # 30.5 is crude, but works ok
        for p in props["right_ips"]:
            xmax_labels.append( date_lo_dt + relativedelta(days=p*30.5) )
        if False:
            ax.hlines(y=props["width_heights"], xmin=xmin_labels, xmax=xmax_labels, color = c)
    #
    # create the jump legend. of we do not have the peaks legend
    if not args.peaks and jumps >= args.jump_count: # in the claim date we might want 3 or so
        line, = ax.plot( date_labels, xx, c, alpha=.5, label=read_fgrps[i] ) #, marker='.' ) # plot thicker/marker
        lines.append( line )
        fgrps.append( read_fgrps[i] )  #these appear in the legend as well, and are printed below
    else:
        ax.plot( date_labels, xx, c, alpha=.5 )
if len(lines) <= args.legend:
    ax.legend(lines, fgrps, prop={'size': 8})
ax.set_title( title_str )
#
if args.ymax:
    ax.set_ylim(None, args.ymax)
#
if not args.anon:
    anno = ax.annotate(param_str, xy=(0, -0.24), xycoords=ax.transAxes)
#
ax.xaxis.set_minor_locator( MonthLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]) )
ax.xaxis.set_major_locator( YearLocator() )
ax.xaxis.set_major_formatter( DateFormatter('%Y-%m') )
fig.autofmt_xdate()
plt.tight_layout()
if args.anon:
    #ax.get_xaxis().set_visible(False) # dates are OK
    #ax.get_yaxis().set_visible(False)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_yticklabels(empty_string_labels)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_xticklabels(empty_string_labels)
    try:
        ax.get_legend().remove()
    except AttributeError:
        pass # if no legend, "AttributeError: 'NoneType' object has no attribute 'remove'"
if chunks > 0:
    pngfile = "ts6plot_"+args.date_type+"_"+date_lo+"-"+date_hi+"_c"+str(chunks)+"_m"+str(args.margin)+".png"
else:
    pngfile = "ts6plot_"+args.date_type+"_"+date_lo+"-"+date_hi+"_fgrp"+str(fname_fgrp)+"_p"+args.start_month+"_m"+str(args.months)+"_"+args.date_type+".png"
if os.path.exists( pngfile ):
    os.remove( pngfile )
fig.savefig(pngfile, dpi=300)
print( "Saved", pngfile )
#
print( len(read_fgrps), read_fgrps )
print( "fgrps/legend", ",".join(fgrps) )
#
print( "topn.most_common(8) --------" )
for a,b in topn.most_common(8):
    print( a, b )
print( ",".join( sorted( [ x for (x,y) in topn.most_common(8) ] ) ))
#
print( "peaks_hi.most_common(8) --------" )
for a, b in peaks_hi.most_common(8):
    print( a, b )
print( ",".join( sorted( [ x for (x,y) in peaks_hi.most_common(8) ] ) ))
#
print( "Intersection (all) --------" )
set_topn = set( topn )
set_peaks_hi = set( peaks_hi )
inters = set_topn & set_peaks_hi
print( len(inters), ",".join(sorted(inters)) )
#
if args.adf:
    print( "ADF Stats --------" )
    adf_stats_df = pd.DataFrame( adf_stats, columns=["FGRP4", "ADFStat", "pval", "Lags", "Observations", "stationary", "1%"] )
    print( adf_stats_df[(adf_stats_df["stationary"]==False)] )
    print( "-" * 8 )
    print( adf_stats_df[(adf_stats_df["stationary"]==True)] )
    #print( adf_stats_df )
    print( "-" * 8 )

if args.info:
    plt.show(block=args.block)
    sys.exit(1)

# ALSO
# for each of the fgrps, we could take the average or a boxplot of the number of claims
# per month.
    
# ----

if args.kmeans_algo == 0:
    k_title = "Euclidean $k$-means"
    f_title = "euclidian"
    km = TimeSeriesKMeans(n_clusters=num, verbose=True, random_state=seed)
    y_pred = km.fit_predict(X)
    print( y_pred )
elif args.kmeans_algo == 1:
    k_title = "DBA"
    f_title = "DBA_k_means"
    km = TimeSeriesKMeans(n_clusters=num, n_init=2, metric="dtw", verbose=True,
                              max_iter_barycenter=10, random_state=seed)
    y_pred = km.fit_predict(X)
    print( y_pred )
elif args.kmeans_algo == 2:
    k_title = "Soft-DTW k-means"
    f_title = "soft_DTW"
    km = TimeSeriesKMeans(n_clusters=num, metric="softdtw", metric_params={"gamma_sdtw": .01}, #was 0.01
                          verbose=True, random_state=seed, n_init=2, #default=1
                          max_iter_barycenter=1000 #default=100
    )
    y_pred = km.fit_predict(X)
    print( y_pred )
else:
    k_title = "KShape"
    f_title = "KShape"
    km = KShape(n_clusters=num, verbose=True, random_state=seed)
    y_pred = km.fit_predict(X)
    print( y_pred )
    
plt.figure()
for yi in range(num):
    plt.subplot(num, 1, yi + 1)
    for xx in X[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    if y_min < 100 and y_max > -100:
        plt.ylim(y_min-1, y_max+1)
    if yi == 0:
        plt.title( k_title )
    if args.anon:
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.tight_layout()

PNGFILE="tscluster_"+f_title+"_fgrp"+str(fgrp)+"_k"+str(args.kmeans_algo )+".png"
if os.path.exists( PNGFILE ):
    os.remove( PNGFILE )
plt.savefig(PNGFILE, dpi=300)
print( "Saved", PNGFILE )

for c in range(0, num):
    N = [i for i,x in enumerate(y_pred) if x == c]
    # N contains indices of the predicted classes, which correlate to read_fgrps
    print( c, [read_fgrps[i] for i in N] ) # if just one, it is an outlier
    
# ----

plt.show(block=args.block)

'''
for x in 1014 1113 1115 1122 1124 1126 1129 1142 1143 1144 1149 1152 1157 1162 1163 1167 1223 1413 1414 1415 1416 1512 1514 1515 1519 1613 1614 1615 1619 1632 1792 1832 1841 1871 1891 1914 1982 2102 2105 2108 2111 2112 2115 2116 2117 2121 2125 2126 2129 2131 2132 2133 2134 2139 2141 2142 2143 2144 2145 2146 2148 2149 2151 2152 2153 2154 2155 2158 2161 2162 2163 2164 2165 2166 2167 2168 2169 2171 2172 2181 2184 2211 2212 2213 2214 2215 2219 2221 2222 2228 2229 2231 2303 2319 2331 2334 2337 2341 2342 2346 2348 2349 2361 2366 2371 2372 2373 2374 2375 2379 2384 2389 2431 2439 2441 2446 2459 2512 2513 2515 2516 2521 2522 2523 2524 2525 2527 2528 2529 2532 2533 2535 2538 2539 2542 2543 2545 2547 2548 2549 2551 2552 2561 2562 2563 2564 2565 2571 2572 2579 2581 2582 2584 2585 2586 2587 2588 2589 2611 2612 2614 2615 2616 2617 2618 2619 2621 2626 2627 2628 2629 2631 2632 2633 2634 2636 2637 2639 2651 2652 2663 2691 2711 2715 2731 2732 2741 2811 2841 2842 2846 2849 2861 2862 2869 2931 2932 2933 2934 2935 2939 2991 3001 3002 3011 3019 3111 3112 3113 3119 3121 3131 3139 3211 3212 3224 3241 3311 3318 3331 3334 3339 3341 3349 3513 3514 3519 3521 3522 3525 3526 3527 3531 3535 3538 3541 3552 3561 3565 3569 3571 3621 3622 3624 3629 3631 3634 3635 3637 3638 3639 3640 3641 3643 3644 3645 3646 3648 3649 3651 3662 3665 3669 3681 3689 3711 3712 3713 3714 3716 3719 3721 3722 3723 3725 3729 3731 3733 3734 3735 3739 3741 3742 3745 3749 3751 3752 3753 3754 3757 3758 3759 3761 3765 3769 3811 3819 3822 3829 3835 3843 3852 3861 3862 3864 3865 3869 3872 3902 3903 3911 3929 3931 3949 3952 3953 3971 3972 3982 3989 4111 4112 4113 4114 4117 4122 4133 4134 4135 4136 4137 4138 4139 4144 4149 4212 4213 4219 4249 4311 4312 4313 4314 4315 4316 4317 4319 4321 4322 4323 4324 4325 4326 4327 4328 4329 4341 4343 4371 4372 4376 4378 4379 4384 4453 4511 4513 4514 4515 4531 4532 4533 4601 4602 4609 4611 4619 4651 4653 4654 4655 4656 4657 4658 4659 4661 4663 4664 4665 4666 4669 4671 4684 4811 4814 4816 4821 4824 4825 4832 4839 4911 4912 4953 5111 5112 5113 5114 5115 5116 5117 5119 5121 5123 5124 5126 5129 5131 5134 5141 5211 5213 5222 5241 5514 5516 5611 5612 5614 5617 5618 5619 5621 5622 5629 5631 5633 5635 5639 5651 5653 5654 5655 5659 5711 5731 5739 5911 5921 5929 5931 5939 6102 6112 6113 6119 6121 6122 6125 6126 6129 6411 6412 6413 6414 6419 6421 6422 6424 6428 6429 6431 6434 6436 6438 6439 6451 6452 6453 6454 6455 6456 6457 6459 6511 6521 6522 6523 6525 6527 6551 6553 6554 6555 6559 6562 6571 6999 7111 7112 7113 7114 7116 7118 7119 7121 7123 7131 7149 7171 7173 7211 7212 7213 7214 7219 7221 7222 7229 7242 7252 7261 7262 7269 7271 7273 7281 7422 7611 7612 7613 7614 7621 7622 7629 7641 7644 7645 7647 7648 7661 7711 7712 7717 7721 7722 7731 7732 7735 7736 7739 7741 7761 7762 7999 8013 8101 8102 8109 8111 8112 8114 8117 8121 8126 8131 8136 8154 8172 8181 8182 8189 8211 8212 8213 8219 8231 8232 8241 8251 8254 8259 8271 8311 8312 8315 8318 8321 8341 8342 8343 8344 8345 8349 8351 8352 8361 8365 8412 8415 8417 8431 8432 8433 8434 8435 8441 8444 8445 8451 8454 8457 8461 8462 8463 8469 8481 8511 8521 8525 8526 8529 8552 8554 8556 8561 8562 8579 8611 8615 8631 8639 8655 8659 8712 8715 8721 8724 8731 8732 8733 8734 8739 8741 8742 8743 8744 8746 8747 8748 8752 8761 8771 8781 8811 8812 8821 8825 8841 8845 8847 8851 8912 8913 8915 8916 8917 8918 8919 8921 8961 8962 8966 8969 8971 8979 8995 8999 9131 9218 9219 9221 9222 9224 9225 9227 9229 9819 9862 9939 9999 ; do python3 plot_claims_proddate4.py -f $x -p 2011-01 -m108;done

for x in 3001 3002 3011 3019 3111 3112 3113 3119 3121 3131 3139 3211 3212 3224 3241 3311 3318 3331 3334 3339 3341 3349 3513 3514 3519 3521 3522 3525 3526 3527 3531 3535 3538 3541 3552 3561 3565 3569 3571 3621 3622 3624 3629 3631 3634 3635 3637 3638 3639 3640 3641 3643 3644 3645 3646 3648 3649 3651 3662 3665 3669 3681 3689 3711 3712 3713 3714 3716 3719 3721 3722 3723 3725 3729 3731 3733 3734 3735 3739 3741 3742 3745 3749 3751 3752 3753 3754 3757 3758 3759 3761 3765 3769 3811 3819 3822 3829 3835 3843 3852 3861 3862 3864 3865 3869 3872 3902 3903 3911 3929 3931 3949 3952 3953 3971 3972 3982 3989; do python3 plot_claims_proddate5.py -f $x -p 2011-01 -m120;done

'''
