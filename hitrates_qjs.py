#!/usr/bin/env python3
#
import re
import sys, os
import argparse
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib as mpl
mpl.use("Qt5Agg") #TkAgg crashes
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns
import itertools
from matplotlib.dates import MonthLocator, DateFormatter, YearLocator

from read_data import *
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import statsmodels.api as sm
from sklearn import linear_model

pd.options.display.width = 140

mpl_colours = [ "#1f77b4", "#ff7fe0", "2ca02c", "#d62728", "#9467bd",
                "#8c564b", "e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
cat20_colours = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
                 "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
                 "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", 
                 "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

plot_size = (10,6) #(12, 8)
plot_dpi = 144
#plot_prefix = "PROGPRES"
plot_prefix = "ARISE_"

'''
PLT:
  ( (repd_)combined.index > '2014-06-01' )
PLT1:
  ( full data )
PLT2:
  > 2014-06-01, weekly
'''

def scatter(foo, ax4, title, xlims, ylims):
    # scatter plot on days/hitrate
    # Filter out extreme ones?
    #foo     = foo[ (foo['hitrate'] >= 0) & (foo['hitrate'] < 4) ]
    info    = "hitrate: [ " + str(args.hitrate_min) +", "+ str(args.hitrate_max) + " )"
    # Also filter length?
    foo     = foo[ (foo['days'] > 0) & (foo['days'] < 2*365) ]
    info   += "\ndays: [ " + str(args.days_min) +", "+ str(args.days_max) + " )"
    colours = np.where(foo.hitrate < 1, 'g', 'r')
    #foo["c"] = "r" # default colour
    #foo["c"][foo["hitrate"] < 1] = 'g'
    #foo["c"][foo["days"] <= 14] = 'b'
    good_count = foo['hitrate'][foo['hitrate'] < 1.0 ].count()
    bad_count  = foo['hitrate'][foo['hitrate'] >= 1.0 ].count()
    #print( good_count, bad_count )
    ax4.scatter( x=foo['days'], y=foo['hitrate'], c=colours ) #c=foo["c"] )
    ax4.set_ylim( *ylims )
    ax4.set_xlim( *xlims )
    ax4.set_title( title )
    ax4.grid( b=True, axis="both", linestyle="dotted" )
    ax4.legend(fontsize='x-small', labelspacing=0.2, frameon=True)
    ax4.set_ylabel( "hitrate" )
    ax4.set_xlabel( "length of QJ in days" )
    #print( foo['days'].values,foo['hitrate'].values )
    '''
    try:
        z = np.polyfit( foo['days'].values,foo['hitrate'].values, 1 )
        p = np.poly1d( z )
        ax4.plot( foo['days'].values, p(foo['days'].values), "b", alpha=0.5 )
        #print( z, p )
    except TypeError:
        pass
    '''
    #print( np.corrcoef( foo['days'].values,foo['hitrate'].values ) )
    #sns.jointplot("days", "hitrate", data=foo, kind='reg', ylim=(-1,5))
    gc_str = "{0:\u2002>3d}".format( good_count ) # U+2002 is better than default
    bc_str = "{0:\u2002>3d}".format( bad_count )
    if good_count >= bad_count:
        ax4.text(0.9, 0.9,  gc_str, fontsize=14, transform=ax4.transAxes, color='g')
        ax4.text(0.9, 0.84, bc_str, fontsize=14, transform=ax4.transAxes, color='r')
    else:
        ax4.text(0.9, 0.9,  bc_str, fontsize=14, transform=ax4.transAxes, color='r')
        ax4.text(0.9, 0.84, gc_str, fontsize=14, transform=ax4.transAxes, color='g')
    ax4.text(0.64, 0.9, info, fontsize=14, transform=ax4.transAxes, color='k')
    return good_count, bad_count

parser = argparse.ArgumentParser()
parser.add_argument( '-a', "--anon", action='store_true', default=False, help='No labels on axes' )
parser.add_argument( '-b', "--block", action='store_true', default=False, help='Keep showing the plot until closed' )
parser.add_argument( '-f', '--fgrps', action='append', help='Specify FGRP4s')
parser.add_argument( "-g", "--gap", type=int, default=270, help='Gap in days after QJ close' )
parser.add_argument( '-m', "--movie_files", action='store_true', default=False, help='Create movie files' )
parser.add_argument( '-y', "--ym_forced", type=float, default=None, help='Force y-axis max value' )

parser.add_argument( "--hitrate_min", type=float, default=0, help='Minimum hitrate on plots' )
parser.add_argument( "--hitrate_max", type=float, default=9.9, help='Maximum hitrate on plots' )

parser.add_argument( "--days_min", type=int, default=0, help='Minimum QJ length on plots' )
parser.add_argument( "--days_max", type=int, default=3*365, help='Maximum QJ length on plots' )

parser.add_argument( "--keep_zeroes", action='store_true', default=False, help='Keep time spans with 0 claims' )

args = parser.parse_args()

colours = ["#0049E5", "#00C3E1", "#00D906", "#6FD500", "#D2C300", "#2100BF", "#CE4E00", "#CA0022",
           "#C6008F", "#8D00C2", "#00DD80",
           "#6F00E5","#0002E1","#0072DD","#00D9D5","#00D566",
           "#04D200","#6CCE00","#CAC400","#C65D00", "#C20006","#BF0066"]
#colours = cat20_colours[4:]

foo = DATA( verbose=False )
foo.read_full()
'''
for f in [ "C Value (QJ)", "F Value (QJ)", "S Value (EUR) (QJ)", "S value corrected Eur", "N Value (QJ)", 
           "Root Cause Responsibility BA/BU", "Problem Cause Initiation BA/BU" ]:
    print( f )
    print( foo.qjs[f].value_counts() )
    print()
sys.exit(1)
'''
if not args.fgrps:
    args.fgrps = [3111] #foo.fgrp4_list # [3111, 2846, 8752, 2584, 2581, 3651, 5931, 8746]
    args.fgrps = foo.fgrp4_list

print( "----" )
print( "Examining", len(args.fgrps), "function groups." )
print( "QJ length min/max in days:", args.days_min, args.days_max )
print( "Hitrate min/max:", args.hitrate_min, args.hitrate_max )
print( "Hitrate 'gap':", args.gap )
print( "----" )

qj_info = pd.DataFrame( columns=["fgrp4", "fg", "qj_open", "qj_close", "days",
                                 "cr_before", "cr_during", "cr_after", "hitrate",
                                 "cc_before", "cc_during", "cc_after"] )
for x in args.fgrps:
    x = int(x)
    qjs, claims, combined, repd_combined = foo.select_fgrp( x )
    #if claims.empty:
    #    continue
    sep( str(x) )
    #print( "fgrp4", x )
    #
    qjs = foo.select_qjs_fgrp( x )
    #qjs = qjs[ (qjs.qj_open > '2014-06-01') ]
    print( "qjs.shape", qjs.shape )
    
    # Plot the hitrate lines for the QJs related to this
    # function group
    rows = []
    for index, row in qjs.iterrows():
        qj_0   = row["qj_open"]
        qj_1   = row["qj_close"]
        fe     = row["f_effect"]
        me     = row["m_effect"]
        fg     = int(row["fg"])
        dt_t   = timedelta(days=90)
        dt_gap = timedelta(days=args.gap) # time to wait after closing Qj for 90 days
        if  qj_0 == -1 or qj_1 == -1:
            print( "SKIP -1 DAY:", row["qj_id"], "/", x, days )
            continue
        days   = qj_1 - qj_0
        days   = int(str(days).split()[0]) # +1 removed, 0 days means "quick fix"
        if  not (args.days_min <= days < args.days_max):
            print( "SKIP DAYS LIMIT:", row["qj_id"], "/", x, days )
            continue
        dt_0   = qj_0 - dt_t  # start of days period before QJ open
        dt_1   = qj_1 + dt_t  # end of days period after QJ close
        dt_1s  = qj_1 + dt_gap  # start of period affter gap after closing
        dt_1e  = dt_1s + dt_t  # end of days period after QJ close
        #print( dt_0, qj_0, date_diff(qj_0, qj_1), qj_1, dt_1 )
        qq = select_between_dates( combined, dt_0, qj_0 ) # NB combined is only about this fgrp4!
        #count_before = len( qq ) # len(qq) is days with claims, not sum of claim count
        if qq.empty:
            count_before = 0
        else:
            count_before = qq["CLAIMCOUNT"].sum() # len(qq) is days with claims, not sum of claim count
        mean_before, ave = calculate_hitrate( qq, 'TOTNORMALISED' )
        #print( mean_before, ave )
        #
        qq = select_between_dates( combined, qj_0, qj_1 )
        # qq_filename = "fgrp"+str(x)+"_qj"+str(index)+"_during.csv"
        # print( "SAVING:", qq_filename  )
        # qq.to_csv( qq_filename, index=True, sep=";", float_format='%.4f' )
        #count_during = len( qq )
        if qq.empty:
            count_during = 0
        else:
            count_during = qq["CLAIMCOUNT"].sum()
        mean_during, ave = calculate_hitrate( qq, 'TOTNORMALISED' )
        #
        qq = select_between_dates( combined, dt_1s, dt_1e )
        #count_after = len( qq )
        if qq.empty:
            count_after = 0
        else:
            count_after = qq["CLAIMCOUNT"].sum()
        mean_after, ave = calculate_hitrate( qq, 'TOTNORMALISED' )
        try:
            hitrate = mean_after/mean_before
        except ZeroDivisionError:
            continue
        if not args.keep_zeroes and not all( [count_before, count_during, count_after] ): #this loses 0 day stuff?
            print( "SKIP 0 COUNT:", row["qj_id"], "/", x )
            continue
        #if not all( [ 0 if x < 4 else 1 for x in [count_before, count_during, count_after] ] ):
        #    continue
        #print( count_before, count_during, count_after )
        if args.keep_zeroes or args.hitrate_min <= hitrate < args.hitrate_max:
            rows.append( [x, fg, qj_0, qj_1, days, mean_before, mean_during, mean_after, hitrate,
                          count_before, count_during, count_after] )
        else:
            print( "SKIP HITRATE LIMIT:", row["qj_id"], "/", x, hitrate )
        # From this, prepare data plot which shows average length of QJ per closing year/month
        # add as column?
        #qj_closing_m = qj_1.strftime( "%Y-%m" ) 
        
    fgrp_qjs = pd.DataFrame( rows,
                             columns=["fgrp4", "fg", "qj_open", "qj_close", "days",
                                      "cr_before", "cr_during", "cr_after", "hitrate",
                                      "cc_before", "cc_during", "cc_after"])
    fgrp_qjs.to_csv( "fgrp"+str(x)+"_qjs.csv", index=True, sep=";", float_format='%.4f' )
    qj_info = qj_info.append( fgrp_qjs,
                              ignore_index=True
    )   
    rows = []

print( qj_info )

print( "SAVING:", "qj_info.csv" )
qj_info.to_csv( "qj_info.csv", index=True, sep=";", float_format='%.4f' )

# Average length in closing month (or other)
# ------------------------------------------
ys_rng = pd.date_range( start="2012-01", end="2019", freq='MS') # or 'MS' and 'M'
ye_rng = pd.date_range( start="2012-02", end="2019", freq='MS') # or 'AS' and 'A'
ye_rng = ye_rng - timedelta(days=1) # make it last day of previous month
#
# Closing dates in the range
# --------------------------
rows = []
for ys,ye in zip(ys_rng, ye_rng):
    #print( str(ys)[0:10], "--", str(ye)[0:10], end=' ' )
    sub_qjs = qj_info[ (qj_info["qj_close"] >= ys) & (qj_info["qj_close"] <= ye) & (qj_info["hitrate"] > 0) ]
    sub_qjs = sub_qjs.dropna()
    if sub_qjs.empty:
        #print( "" ) #force linefeed after daterange if emtpy
        rows.append( [ys, 0] )
        continue
    #print( "{0:4n}".format( sub_qjs['days'].mean() ) )
    rows.append( [ys, sub_qjs['days'].mean()] )
qjs_ave_days = pd.DataFrame( rows, columns=["month", "ave days"])
#print( qjs_ave_days )
fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(12,8)) 
#qjs_ave_days.plot( y="ave days", kind="bar", ax=ax5 )
ax5.bar( qjs_ave_days["month"], qjs_ave_days["ave days"], width=14 )
ax5.set_title( "Average QJ length in closing month" )
if args.anon:
    labels = [item.get_text() for item in ax5.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    ax5.set_yticklabels(empty_string_labels)
    labels = [item.get_text() for item in ax5.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax5.set_xticklabels(empty_string_labels)
#
fig5.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
#
#ax5.xaxis.set_minor_locator(MonthLocator())
#ax5.xaxis.set_minor_formatter(DateFormatter('%m'))
#ax5.xaxis.set_major_locator(YearLocator())
#ax5.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
#
#bin_xticks = ax5.get_xticks()
#ax5.set_xticks( bin_xticks[::1] )
#
#sns.barplot( x=qjs_ave_days["month"], y=qjs_ave_days["ave days"] )
if args.anon:
    PDFFILE="qjs_close_m_ave_anon.png"
else:
    PDFFILE="qjs_close_m_ave.png"
fig5.savefig(PDFFILE, dpi=144)
print( "SAVED", PDFFILE )

fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(12,8), sharex=True) #QJ scatter
scatter( qj_info, ax4, "", (0,2*365), (0, 2) )
PDFFILE="qjs_scatter.png"
fig4.savefig(PDFFILE, dpi=144)
print( "SAVED", PDFFILE )

if not args.movie_files:
    plt.show( block=args.block )
    sys.exit(0)

print( "----" )
    
# Monthly slices over all QJs
# ---------------------------
qj_years = []
ys_rng = pd.date_range( start="2011", end="2019", freq='MS') # or 'MS' and 'M'
ye_rng = pd.date_range( start="2012", end="2020", freq='MS') # or 'AS' and 'A'
ye_rng = ye_rng - timedelta(days=1) # make it last day of previous month
#
##ys_rng = pd.date_range( start="2012", end="2018", freq='AS') # or 'MS' and 'M'
##ye_rng = pd.date_range( start="2013", end="2019", freq='A') # or 'AS' and 'A'
#
#ys_rng = [ "2012-01-01" ]
#ye_rng = [ "2018-01-01" ]

#print( ys_rng, ye_rng  )

# Opening dates in the range
# --------------------------
cnt = 100 # for ffmpeg files, see a few lines below
for ys,ye in zip(ys_rng, ye_rng):
    print( str(ys)[0:10], "--", str(ye)[0:10], end='' )
    sub_qjs = qj_info[ (qj_info["qj_open"] >= ys) & (qj_info["qj_open"] <= ye) & (qj_info["hitrate"] > 0) ]
    sub_qjs = sub_qjs.dropna()
    if sub_qjs.empty:
        print( "" ) #to force the linefeed
        continue
    print("{0:8.4f} {1:8.4f} {2:8.4f} | {3:8.4f} | {4:5.1f} | {5:4n}".format(
        sub_qjs['cr_before'].mean(), sub_qjs['cr_during'].mean(), sub_qjs['cr_after'].mean(), sub_qjs['hitrate'].mean(),
        sub_qjs['days'].mean(),
        len(sub_qjs)
    ), end=' | ' )
    fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(12,8), sharex=True) #QJ scatter
    gd, bd = scatter( sub_qjs, ax4, "Opening date in "+str(ys)[0:10]+"  -  "+str(ye)[0:10], (0, 999), (0, 4) )
    print( gd, bd )
    #fig4.tight_layout()
    PDFFILE=plot_prefix+"qjs_open_rd_"+str(ys)[0:10]+"_"+str(ye)[0:10] + "." + "png"
    fig4.savefig(PDFFILE, dpi=144)
    if True:
        ##ffmpeg -r 4 -start_number 100 -i qj_global_open_%03d.png -vcodec mpeg4 -y movie.mp4
        MFILE=plot_prefix+"qjs_open_rd_"+str(cnt) + ".png"
        fig4.savefig(MFILE, dpi=144)
        #print( "MOVIE FILE:", MFILE )
        cnt += 1
    plt.close(fig4)
print( "ffmpeg -r 4 -start_number 100 -i MOVIE_qjs_open_rd_%03d.png -vcodec mpeg4 -y od_movie.mp4" )

print( "----" )

# Closing dates in the range
# --------------------------
cnt = 100 # for ffmpeg files, see a few lines below
for ys,ye in zip(ys_rng, ye_rng):
    print( str(ys)[0:10], "--", str(ye)[0:10], end='' )
    sub_qjs = qj_info[ (qj_info["qj_close"] >= ys) & (qj_info["qj_close"] <= ye) & (qj_info["hitrate"] > 0) ]
    sub_qjs = sub_qjs.dropna()
    if sub_qjs.empty:
        print( "" ) #to force the linefeed
        continue
    print("{0:8.4f} {1:8.4f} {2:8.4f} | {3:8.4f} | {4:5.1f} | {5:4n}".format(
        sub_qjs['cr_before'].mean(), sub_qjs['cr_during'].mean(), sub_qjs['cr_after'].mean(), sub_qjs['hitrate'].mean(),
        sub_qjs['days'].mean(),
        len(sub_qjs)
    ), end=' | ' )
    fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(12,8), sharex=True) #QJ scatter
    gd, bd = scatter( sub_qjs, ax4, "Closing date in "+str(ys)[0:10]+"  -  "+str(ye)[0:10], (0, 999), (0, 4) )
    print( gd, "-", bd )
    #fig4.tight_layout()
    PDFFILE=plot_prefix+"qjs_close_rd_"+str(ys)[0:10]+"_"+str(ye)[0:10] + "." + "png"
    fig4.savefig(PDFFILE, dpi=144)
    if True:
        ##ffmpeg -r 4 -start_number 100 -i qj_global_open_%03d.png -vcodec mpeg4 -y movie.mp4
        MFILE=plot_prefix+"qjs_close_rd_"+str(cnt) + ".png"
        fig4.savefig(MFILE, dpi=144)
        #print( "MOVIE FILE:", MFILE )
        cnt += 1
    plt.close(fig4)
print( "ffmpeg -r 4 -start_number 100 -i MOVIE_qjs_close_rd_%03d.png -vcodec mpeg4 -y cd_movie.mp4" )

plt.show(block=args.block)
