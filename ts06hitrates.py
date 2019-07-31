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
import sys, os, pickle
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

def get_qjs(the_df, qjs, fgrp):
    #print( "get_qjs/the_df" )
    #print( the_df.head() )
    #print( qjs.head() )
    #
    # Print/calculate QJ info
    #
    qjs_info           = [] #list with all info ?
    claim_rates        = []
    pre_claim_rates_m  = [] #dates (months) used to calculate
    post_claim_rates_m = [] #dates (months) used to calculate
    for index, row in qjs.iterrows():
        qj_0   = row["qj_open"] # real date/time
        qj_1   = row["qj_close"]# real date/time
        if qj_0 == -1 or qj_1 == -1:
            print( "Skipping (-1) QJ", index )
            continue
        days   = qj_1 - qj_0
        days   = int(str(days).split()[0]) # +1 removed, 0 days means "quick fix"
        if  not (0 <= days < 3*365):
            #print( "SKIP DAYS LIMIT:", row["qj_id"], "/", index, days )
            continue
        fe     = row["f_effect"]
        me     = row["m_effect"]
        dt_t   = timedelta(days=90) # now we just have months...
        dt_gap = timedelta(days=90) # time to wait after closing Qj for 90 days
        # six_months = date.today() + relativedelta(months=+6)
        qj_0_m    = qj_0.strftime('%Y-%m') # start date/time month
        qj_1_m    = qj_1.strftime('%Y-%m') # end date/time month
        pre_cr    = [] # pre claim rates
        pre_cr_m  = [] # months used to calculate pre_cr
        post_cr   = [] # post claim rates
        post_cr_m = []
        #
        # Calculate during from monthly values
        # ...
        qj_curr     = qj_0
        during_cr   = []
        during_cr_m = []
        print( "----", index, days )
        #print( index, qj_0, qj_1 )
        while qj_curr < qj_1:
            qj_curr_m = qj_curr.strftime('%Y-%m')
            try:
                during_cr.append( the_df.loc[qj_curr_m] )
                during_cr_m.append( qj_curr.replace(day=1) ) # 12 instead of 1 because bars are 24 days, centered
                print( "CURR", index, qj_curr_m, the_df.loc[qj_curr_m]  )
            except KeyError:
                pass
            qj_curr += relativedelta(months=1)
        try:
            ave_during_cr = sum(during_cr) / len(during_cr)       # <----------- USE THIS BELOW
        except ZeroDivisionError:
            ave_during_cr = 0.0
        #print( during_cr, ave_during_cr )
        # Three months before (maybe use day/time in days, round to months?)
        for d in range(1, args.period_before+1): # start before, not on  
            foo = qj_0 - relativedelta(months=d)
            foo_m = foo.strftime('%Y-%m') # start month
            try:
                #print( "df_loc", foo_m, the_df.loc[foo_m] )
                pre_cr.append( the_df.loc[foo_m] )
                pre_cr_m.append( foo.replace(day=1) ) 
                print( "PRE", index, foo_m, the_df.loc[foo_m] )
            except KeyError:
                pass
        # three months after, skip one
        for d in range(args.gap+1, args.period_after+args.gap+1): #skip one month, so start at next
            foo = qj_1 + relativedelta(months=d)
            foo_m = foo.strftime('%Y-%m') # start month
            try:
                #print( foo_m, the_df.loc[foo_m][idx] )
                post_cr.append(the_df.loc[foo_m])
                post_cr_m.append( foo.replace(day=1) )
                print( "POST", index, foo_m, the_df.loc[foo_m] )
            except KeyError:
                pass
                #print( foo_m )
        try:
            ave_pre_cr = sum(pre_cr)/len(pre_cr)
        except ZeroDivisionError:
            ave_pre_cr = 0
        try:
            ave_post_cr = sum(post_cr)/len(post_cr)
        except ZeroDivisionError:
            ave_post_cr = 0
        #print( index, ave_pre_cr, ave_post_cr )
        claim_rates.append( [qj_0, ave_pre_cr, qj_1, ave_post_cr, days, ave_during_cr] )
        dt_0  = qj_0 - dt_t  # start of days period before QJ open
        dt_1  = qj_1 + dt_t  # end of days period after QJ close
        dt_1s = qj_1 + dt_gap  # start of period affter gap after closing
        dt_1e = dt_1s + dt_t  # end of days period after QJ close
    for i,values in enumerate(claim_rates):
        mean_before = values[1]
        mean_after  = values[3]
        mean_during = values[5]
        qj_open     = values[0]
        qj_close    = values[2]
        days        = values[4]
        if mean_before == 0 or mean_after == 0:
            print( "SKIP, zero means" )
            continue
        hitrate = mean_after / mean_before
        print( "--->", [ i, fgrp, time_str(qj_open), time_str(qj_close), days, flt(mean_before),
                         flt(mean_during), flt(mean_after), flt(hitrate) ] )
        qjs_info.append( [ i, fgrp,
                           qj_open, str(qj_open)[0:7], str(qj_open)[0:4],
                           qj_close, str(qj_close)[0:7], str(qj_close)[0:4],
                           days, mean_before,
                           mean_during, mean_after, hitrate ] )
        # we know this loops in parallel with the months
    return qjs_info

parser = argparse.ArgumentParser()
parser.add_argument( '-a', "--anon", action='store_true', default=False, help='Anonymous axes' )
parser.add_argument( "--autocorr_plot", action='store_true', default=False, help='Plot autocorrelation plots' )
parser.add_argument( '-f', '--fgrp', type=str, default=3, help='Specify FGRP4s')
parser.add_argument( '-F', '--notfgrp', type=str, default="", help='FGRP4s to ignore')
parser.add_argument( '-d', '--date_type', type=str, default="claim", help='Date type, "prod" or "claim" (default)')
#
parser.add_argument( '-g', "--gap", type=int, default=1, help='Gap after QJ closing month in months' )
parser.add_argument( "--period_after", type=int, default=3, help='Number ofmonths to calculate claim rate on' )
parser.add_argument( "--period_before", type=int, default=3, help='Number ofmonths to calculate claim rate on' )
parser.add_argument( "--min_qj_len", type=int, default=1, help='Minimum length of the QJ ( >= 1 )' )
parser.add_argument( "--max_qj_len", type=int, default=1096, help='Maximum length of the QJ ( < 1096 )' )
#
parser.add_argument( "--polyfit", action='store_true', default=False, help='Fit line' )
parser.add_argument( '-k', "--kmeans_algo", type=int, default=0, help='k-Means algorithm' )
parser.add_argument( '-l', "--legend", type=int, default=10, help='Plot legend if less than this (10)' )
parser.add_argument( '-m', "--months", type=int, default=120, help='Plot this many months' ) # from datafile name
parser.add_argument( '-M', "--margin", type=float, default=0.1, help='Margin for chunkifier (0.1)' ) 
parser.add_argument( '-p', "--start_month", type=str, default="2010-01", help='Start month' )
parser.add_argument( '-y', "--ymax", type=float, default=6, help='Y-max for axis' )
parser.add_argument( '-x', "--xmax", type=float, default=1096, help='X-max for axis' )
parser.add_argument(       "--xmin", type=float, default=0, help='X-min for axis' )
parser.add_argument( "--dl", type=str, default="2016-01", help='Date lo' )
parser.add_argument( "--dh", type=str, default="2018-01", help='Date hi' )
args = parser.parse_args()

args_fgrp = str(args.fgrp)

from shlex import quote
invocation = "python3 " + ' '.join(quote(s) for s in sys.argv)
with open( "invocation_ts06hr_log.txt", "a") as f:
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

def time_str(t):
    return (str(t)[0:10]).replace("-", "")

def flt(f):
    return "{:.2f}".format(f)

ms = args.months
sd = args.start_month
#
X = np.asarray([]) # np array

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

# ----

if os.path.exists( "foo.pickle" ):
    with open("foo.pickle", "rb") as f:
        foo = pickle.load( f )
else:
    foo = DATA(verbose=True)
    foo.read_full()
    with open("foo.pickle", "wb") as f:
        pickle.dump( foo, f )

# ----
        
fgrp_list    = args.fgrp.split(",")
notfgrp_list = args.notfgrp.split(",")
y_min =  100
y_max = -100
date_lo = args.dl
date_hi = args.dh
title_str = "Claims on "+args.date_type+" date ("+date_lo+" -- "+date_hi+")"
param_str = ""
    
date_lo_dt   = datetime.strptime(date_lo, '%Y-%m')
date_hi_dt   = datetime.strptime(date_hi, '%Y-%m')
r            = relativedelta( date_hi_dt, date_lo_dt )
delta_months = r.months + (r.years*12)
date_labels  = []

for m in range( 0, delta_months ): 
    prod_dt = date_lo_dt + relativedelta(months = m)
    #prod_date     = str(prod_dt)[0:7] #YYYY-MM
    date_labels.append( prod_dt )

read_fgrps = []
all_qjs_info = []
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
        #my_ts = my_ts[ (my_ts.index >= date_lo) & (my_ts.index < date_hi) ]
        my_ts = my_ts.fillna(0)
        #print( my_ts )
        non_zero = my_ts.astype(bool).sum(axis=0)
        if non_zero < 1:
            print( "SKIP", fn )
            continue
        print( fn )
        print( fgrp, my_ts.max() )
        read_fgrps.append( fgrp ) #the ones which were ok

        # CALCULATE HITRATES ACCORDING TO QJS
        #qjs, claims, combined, repd_combined = foo.select_fgrp( int(fgrp) ) # IS THIS NEEDED?
        qjs = foo.select_qjs_fgrp( fgrp )
        #print( qjs )
        #qjs = qjs[ (qjs.qj_open > args.prod_date) ]
        qjs_info = get_qjs(my_ts, qjs, fgrp)
        for q in qjs_info:
            #print( q )
            # q[8] is num of days
            if args.min_qj_len <= q[8] < args.max_qj_len:
                all_qjs_info.append( q )
        
    except FileNotFoundError:
        print( "Not found", fn )
        continue
    X1 = my_ts.values.reshape(1,-1) #make it into one row
    X1 = np.nan_to_num(X1)
    try:
        X = np.concatenate((X, X1))
    except ValueError: # first time when X is empty
        X = X1
        
print( X.shape )
#print( X )
seed = 42
sz   = X.shape[1]-1

print( "-" * 40 )

# ----

#for i,xx in enumerate(X): # this should be in fgrp order
#    print ( xx )
print( len(read_fgrps), read_fgrps )

all_qjs_df = pd.DataFrame( all_qjs_info, columns=["idx", "fgrp4", "qj_open", "qj_open_ym", "qj_open_y",
                                                  "qj_close", "qj_close_ym", "qj_close_y",
                                                  "days",
                                                  "cr_before", "cr_during", "cr_after", "hitrate"] )

print( all_qjs_df )

# Here we can have statistics on all QJs closed in 2012, or in a certain months, etc.
# In plotting we could have a different symbol for each year...
all_years = sorted(pd.unique(all_qjs_df["qj_close_y"].values)) # or qj_close_ym
boxplotdata   = []
boxplotlabels = []
print( all_years )
for yi,y in enumerate(all_years):
    qjs_y = all_qjs_df[ (all_qjs_df["qj_close_y"]==y) ]
    boxplotdata.append( all_qjs_df[ (all_qjs_df["qj_close_y"]==y) ]["days"].values )
    if not args.anon:
        boxplotlabels.append( str(y) )
    else:
        boxplotlabels.append( str(yi) )
    print( "{} {} {:3n} {:.4f}".format(y, args_fgrp, qjs_y.shape[0], qjs_y["hitrate"].mean()) ) # print the mean
#print( all_qjs_df[ (all_qjs_df["qj_close_y"]=="2015") ] )

param_str = "b"+str(args.period_before)+"g"+str(args.gap)+"a"+str(args.period_after)
param_str += "_mnl"+str(args.min_qj_len)+"_mxl"+str(args.max_qj_len)
CSVFILE="allqjs_"+param_str+"_"+str(args.start_month)+"_fgrp"+args_fgrp+"_"+str(args.months)+"m_"+args.date_type+".csv"
if os.path.exists( CSVFILE ):
    os.remove( CSVFILE )
all_qjs_df.to_csv(CSVFILE, sep=";")#, float_format='%.4f')
print( "SAVED", CSVFILE )

# ----

for x,y in zip(boxplotdata, boxplotlabels):
    try:
        k2, p = stats.normaltest(x)
        if p < 0.001:
            norm = "not normal (a=0.001)"
        else:
            norm = "normal (a=0.001)"
        print( "{:5s} {:3n} {:7.4f} {:7.4f} {}".format(y, len(x), k2, p, norm) )
    except ValueError:
        print( "skewtest is not valid with less than 8 samples" )

# ----

fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
#ax0.hist( all_qjs_df["days"], bins=int(args.xmax), rwidth=1)#[0, 15, 31, 62, 93] ) "auto"
#ax0.boxplot( all_qjs_df["days"], vert=False, meanline=True, showmeans=True)
boxplotdata.append( all_qjs_df["days"].values )
boxplotlabels.append( "all" )
ax0.boxplot( boxplotdata, vert=False,
             meanline=True, showmeans=True,
             labels=boxplotlabels )
ax0.set_ylabel( "Year" )
if not args.anon:
    ax0.set_xlabel( "length of QJ in days ["+str(args.min_qj_len)+", "+str(args.max_qj_len)+"[" )
else:
    ax0.set_xlabel( "length of QJ in days" )
if args.anon:
    labels = [item.get_text() for item in ax0.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax0.set_xticklabels(empty_string_labels)
    PNGFILE="qj_box_"+param_str+"_"+str(args.start_month)+"_fgrp"+args_fgrp+"_"+str(args.months)+"m_"+args.date_type+"_anon.png"
else:
    PNGFILE="qj_box_"+param_str+"_"+str(args.start_month)+"_fgrp"+args_fgrp+"_"+str(args.months)+"m_"+args.date_type+".png"
if os.path.exists( PNGFILE ):
    os.remove( PNGFILE )
fig0.savefig(PNGFILE, dpi=300)
print( "Saved", PNGFILE )

# ----

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
lines = []
fgrps = []
topn  = Counter()
peaks_hi  = Counter()
peaks_cwt = []
adf_stats = []

#ax.set_yscale("log")
#ax.set_xscale("log")
#ylims = (pow(10,-3), pow(10,2))
ylims = (0, args.ymax)
xlims = (args.xmin, args.xmax)
if not args.anon:
    title = "Hitrates / "+args_fgrp
else:
    title = "Hitrates"
    
info    = "hitrates"
# Also filter length?
all_qjs_df     = all_qjs_df[ (all_qjs_df['days'] > 0) & (all_qjs_df['days'] < 3*365) ]
##info   += "\ndays: [ " + str(args.days_min) +", "+ str(args.days_max) + " )"

# Print per year, se we can have different marker symbols.
markers = [ "o", ".", "+", "d", "s", "p", "1", "2", "3", "4", "v", "^", "<", ">" ]
col="qj_close_y"
all_years = sorted(pd.unique(all_qjs_df[col].values)) # or qj_close_ym
l_sym = []
l_txt = []
scs   = []
for iy,y in enumerate(all_years):
    qjs_y_df = all_qjs_df[ (all_qjs_df[col]==y) ]
    if not args.anon:
        colours  = np.where(qjs_y_df.hitrate < 1, 'g', 'r')
    else:
        colours  = "darkgrey"
    sc = ax.scatter( x=qjs_y_df['days'], y=qjs_y_df['hitrate'], c=colours, marker=markers[iy % len(markers)] )
    l_sym.append(sc)
    if not args.anon:
        l_txt.append( str(y) )
    else:
        l_txt.append( str(iy) )
ax.legend(l_sym, l_txt, scatterpoints=1, prop={'size': 8}, bbox_to_anchor=(1.1, 1.0))


# Without the loop above we can plot all_qjs_df simultaneously---
#colours = np.where(all_qjs_df.hitrate < 1, 'g', 'r')
##all_qjs_df["c"] = "r" # default colour
##all_qjs_df["c"][all_qjs_df["hitrate"] < 1] = 'g'
##all_qjs_df["c"][all_qjs_df["days"] <= 14] = 'b'
##print( good_count, bad_count )
#ax.scatter( x=all_qjs_df['days'], y=all_qjs_df['hitrate'], c=colours ) #c=all_qjs_df["c"] )
# ---

good_count = all_qjs_df['hitrate'][all_qjs_df['hitrate'] < 1.0 ].count()
bad_count  = all_qjs_df['hitrate'][all_qjs_df['hitrate'] >= 1.0 ].count()

ax.set_ylim( *ylims )
ax.set_xlim( *xlims )
ax.set_title( title )
ax.grid( b=True, axis="both", linestyle="dotted" )
#ax.legend(fontsize='x-small', labelspacing=0.2, frameon=True)
ax.set_ylabel( "hitrate" )
if not args.anon:
    ax.set_xlabel( "length of QJ in days ["+str(args.min_qj_len)+", "+str(args.max_qj_len)+"[" )
else:
    ax.set_xlabel( "length of QJ in days" )
#print( all_qjs_df['days'].values,all_qjs_df['hitrate'].values )
if args.polyfit:
    try:
        z = np.polyfit( all_qjs_df['days'].values,all_qjs_df['hitrate'].values, 1 )
        p = np.poly1d( z )
        ax.plot( all_qjs_df['days'].values, p(all_qjs_df['days'].values), "b", alpha=0.5 )
        print( "POLYFIT", z, p )
    except TypeError:
        pass
#print( np.corrcoef( all_qjs_df['days'].values,all_qjs_df['hitrate'].values ) )
#sns.jointplot("days", "hitrate", data=all_qjs_df, kind='reg', ylim=(-1,5))

if args.anon:
    labels = [item.get_text() for item in ax.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_yticklabels(empty_string_labels)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_xticklabels(empty_string_labels)
    
gc_str = "{0:\u2002>3d}".format( good_count ) # U+2002 is better than default
bc_str = "{0:\u2002>3d}".format( bad_count )
if not args.anon:
    if good_count >= bad_count:
        ax.text(0.9, 0.9,  gc_str, fontsize=14, transform=ax.transAxes, color='g')
        ax.text(0.9, 0.84, bc_str, fontsize=14, transform=ax.transAxes, color='r')
    else:
        ax.text(0.9, 0.9,  bc_str, fontsize=14, transform=ax.transAxes, color='r')
        ax.text(0.9, 0.84, gc_str, fontsize=14, transform=ax.transAxes, color='g')
#ax.text(0.64, 0.9, info, fontsize=14, transform=ax.transAxes, color='k')

if args.anon:
    PNGFILE="qj_scatter_"+param_str+"_"+str(args.start_month)+"_fgrp"+args_fgrp+"_"+str(args.months)+"m_"+args.date_type+"_anon.png"
else:
    PNGFILE="qj_scatter_"+param_str+"_"+str(args.start_month)+"_fgrp"+args_fgrp+"_"+str(args.months)+"m_"+args.date_type+".png"
if os.path.exists( PNGFILE ):
    os.remove( PNGFILE )
fig.savefig(PNGFILE, dpi=300)
print( "Saved", PNGFILE )
plt.show(block=True)

'''
for x in 1014 1113 1115 1122 1124 1126 1129 1142 1143 1144 1149 1152 1157 1162 1163 1167 1223 1413 1414 1415 1416 1512 1514 1515 1519 1613 1614 1615 1619 1632 1792 1832 1841 1871 1891 1914 1982 2102 2105 2108 2111 2112 2115 2116 2117 2121 2125 2126 2129 2131 2132 2133 2134 2139 2141 2142 2143 2144 2145 2146 2148 2149 2151 2152 2153 2154 2155 2158 2161 2162 2163 2164 2165 2166 2167 2168 2169 2171 2172 2181 2184 2211 2212 2213 2214 2215 2219 2221 2222 2228 2229 2231 2303 2319 2331 2334 2337 2341 2342 2346 2348 2349 2361 2366 2371 2372 2373 2374 2375 2379 2384 2389 2431 2439 2441 2446 2459 2512 2513 2515 2516 2521 2522 2523 2524 2525 2527 2528 2529 2532 2533 2535 2538 2539 2542 2543 2545 2547 2548 2549 2551 2552 2561 2562 2563 2564 2565 2571 2572 2579 2581 2582 2584 2585 2586 2587 2588 2589 2611 2612 2614 2615 2616 2617 2618 2619 2621 2626 2627 2628 2629 2631 2632 2633 2634 2636 2637 2639 2651 2652 2663 2691 2711 2715 2731 2732 2741 2811 2841 2842 2846 2849 2861 2862 2869 2931 2932 2933 2934 2935 2939 2991 3001 3002 3011 3019 3111 3112 3113 3119 3121 3131 3139 3211 3212 3224 3241 3311 3318 3331 3334 3339 3341 3349 3513 3514 3519 3521 3522 3525 3526 3527 3531 3535 3538 3541 3552 3561 3565 3569 3571 3621 3622 3624 3629 3631 3634 3635 3637 3638 3639 3640 3641 3643 3644 3645 3646 3648 3649 3651 3662 3665 3669 3681 3689 3711 3712 3713 3714 3716 3719 3721 3722 3723 3725 3729 3731 3733 3734 3735 3739 3741 3742 3745 3749 3751 3752 3753 3754 3757 3758 3759 3761 3765 3769 3811 3819 3822 3829 3835 3843 3852 3861 3862 3864 3865 3869 3872 3902 3903 3911 3929 3931 3949 3952 3953 3971 3972 3982 3989 4111 4112 4113 4114 4117 4122 4133 4134 4135 4136 4137 4138 4139 4144 4149 4212 4213 4219 4249 4311 4312 4313 4314 4315 4316 4317 4319 4321 4322 4323 4324 4325 4326 4327 4328 4329 4341 4343 4371 4372 4376 4378 4379 4384 4453 4511 4513 4514 4515 4531 4532 4533 4601 4602 4609 4611 4619 4651 4653 4654 4655 4656 4657 4658 4659 4661 4663 4664 4665 4666 4669 4671 4684 4811 4814 4816 4821 4824 4825 4832 4839 4911 4912 4953 5111 5112 5113 5114 5115 5116 5117 5119 5121 5123 5124 5126 5129 5131 5134 5141 5211 5213 5222 5241 5514 5516 5611 5612 5614 5617 5618 5619 5621 5622 5629 5631 5633 5635 5639 5651 5653 5654 5655 5659 5711 5731 5739 5911 5921 5929 5931 5939 6102 6112 6113 6119 6121 6122 6125 6126 6129 6411 6412 6413 6414 6419 6421 6422 6424 6428 6429 6431 6434 6436 6438 6439 6451 6452 6453 6454 6455 6456 6457 6459 6511 6521 6522 6523 6525 6527 6551 6553 6554 6555 6559 6562 6571 6999 7111 7112 7113 7114 7116 7118 7119 7121 7123 7131 7149 7171 7173 7211 7212 7213 7214 7219 7221 7222 7229 7242 7252 7261 7262 7269 7271 7273 7281 7422 7611 7612 7613 7614 7621 7622 7629 7641 7644 7645 7647 7648 7661 7711 7712 7717 7721 7722 7731 7732 7735 7736 7739 7741 7761 7762 7999 8013 8101 8102 8109 8111 8112 8114 8117 8121 8126 8131 8136 8154 8172 8181 8182 8189 8211 8212 8213 8219 8231 8232 8241 8251 8254 8259 8271 8311 8312 8315 8318 8321 8341 8342 8343 8344 8345 8349 8351 8352 8361 8365 8412 8415 8417 8431 8432 8433 8434 8435 8441 8444 8445 8451 8454 8457 8461 8462 8463 8469 8481 8511 8521 8525 8526 8529 8552 8554 8556 8561 8562 8579 8611 8615 8631 8639 8655 8659 8712 8715 8721 8724 8731 8732 8733 8734 8739 8741 8742 8743 8744 8746 8747 8748 8752 8761 8771 8781 8811 8812 8821 8825 8841 8845 8847 8851 8912 8913 8915 8916 8917 8918 8919 8921 8961 8962 8966 8969 8971 8979 8995 8999 9131 9218 9219 9221 9222 9224 9225 9227 9229 9819 9862 9939 9999 ; do python3 plot_claims_proddate4.py -f $x -p 2011-01 -m108;done

for x in 3001 3002 3011 3019 3111 3112 3113 3119 3121 3131 3139 3211 3212 3224 3241 3311 3318 3331 3334 3339 3341 3349 3513 3514 3519 3521 3522 3525 3526 3527 3531 3535 3538 3541 3552 3561 3565 3569 3571 3621 3622 3624 3629 3631 3634 3635 3637 3638 3639 3640 3641 3643 3644 3645 3646 3648 3649 3651 3662 3665 3669 3681 3689 3711 3712 3713 3714 3716 3719 3721 3722 3723 3725 3729 3731 3733 3734 3735 3739 3741 3742 3745 3749 3751 3752 3753 3754 3757 3758 3759 3761 3765 3769 3811 3819 3822 3829 3835 3843 3852 3861 3862 3864 3865 3869 3872 3902 3903 3911 3929 3931 3949 3952 3953 3971 3972 3982 3989; do python3 plot_claims_proddate5.py -f $x -p 2011-01 -m120;done

'''
