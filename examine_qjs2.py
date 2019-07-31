#!/usr/bin/env python3
#
# (c) pjb 2018/08/23
#
# Calculates some stats over the full QJ population, including those
# without associated claims.
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
import pickle

import seaborn as sns
#sns.set(color_codes=True)

parser = argparse.ArgumentParser()
parser.add_argument( "-d", "--days", type=int, default=100, help='Days in histogram.' )
parser.add_argument( '-r', "--reread", action='store_true', default=False, help="Reread excel files." )
parser.add_argument( '-v', "--verbose", action='store_true', default=False, help="Verbose" )
parser.add_argument( '-a', "--anon", action='store_true', default=False, help="Anonymise" )
args = parser.parse_args()


# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

# Filenames for the data files
qjdate_open  = "Actual NEW Date" # Opening of QJ
qjdate_close = "last MR" # Closing of QJ ?
        
def sep(x):
    #print( "-" * 40, x )
    pass

# -----------------------------------------------------------------------------
# Code
# -----------------------------------------------------------------------------

def get_fg(x):
    try:
        return int(x.split('-')[0])
    except ValueError:
        # Some have "Unknown" in field
        #print( "ERROR", x )
        return 0

sep( "QJ Information" )

if os.path.exists( "foo.pickle" ):
    with open("foo.pickle", "rb") as f:
        foo = pickle.load( f )
else:
    foo = DATA(verbose=True)
    foo.read_full()

qj_dates_all = pd.DataFrame(columns=["qj_open", "qj_close", "days", "f_effect", "m_effect", "m_open"])
qj_index_all = []
print( "len(foo.qjs)", len(foo.qjs) )
for i, row in foo.qjs.iterrows(): # big ...
    od = row[qjdate_open]
    cd = row[qjdate_close]
    fe = row["Forecasted effectiveness %"]
    me = row["Effectiveness %"] # "measured" effectiveness
    qi = row["QJ #"]
    try:
        ds = int(str(cd - od).split()[0]) + 1 #length of QJ
        hash = str(od)+str(cd)+str(fe)+str(me)+str(qi)    
        if hash not in qj_index_all: # assume only one opened on a certain date
            qj_dates_all = qj_dates_all.append( {'qj_open': od, 'qj_close': cd, 'days': ds,
                                                 "f_effect":fe, "m_effect":me,
                                                 'm_open':str(od)[0:7]}, ignore_index=True )
            qj_index_all.append( hash )
    except ValueError:
        pass
qj_dates_all = qj_dates_all.fillna(-1) # -1s for NaNs
print( "qj_dates_all shape, head and tail" )
print( qj_dates_all.shape )
print( qj_dates_all.head() )
print( qj_dates_all.tail() )

#counts = np.bincount( qj_dates_all["days"] )[0:9]
#print( counts )
#fig, ax = plt.subplots()
#ax.bar( range(10), counts, width=1, align='center' )
#ax.set( xticks=range(10), xlim=[-1, 10] )


data = qj_dates_all["days"]
print( data.value_counts() )
fig, ax = plt.subplots( figsize=(8, 5) )
r = args.days 
if r == 0:
    r = len(data)
#r += 1 # because of 0
bins = np.arange(r+1) - 0.5
#print( bins )
ax.hist( data, bins, rwidth=0.8 )
divs = int(r/10)
if divs == 0:
    divs = 1
ax.set_xticks(range(r)[::divs])
ax.set_xlim([-1, r])
ax.set_title( "QJ length distribution" )
ax.set_ylabel( 'Count' )
ax.set_xlabel( 'Days' )
if args.anon:
    labels = [item.get_text() for item in ax.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_xticklabels(empty_string_labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_yticklabels(empty_string_labels)
#
#ml = mpl.ticker.AutoMinorLocator(5)
#ax.yaxis.set_minor_locator(ml)
#
if r < 101:
    locmin = mpl.ticker.LinearLocator( numticks=r+2 )
    #locmin.MAXTICKS = 10000
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    #for label in ax.xaxis.get_ticklabels(minor=True)[::2]:
    #    label.set_visible(False)
#
fig.tight_layout()
if not args.anon:
    fn = "qj_dates_all_d"+str(r)+".png"
else:
    fn = "qj_dates_all_d"+str(r)+"_anon.png"
fig.savefig(fn, dpi=144)
print( "SAVED", fn )

#sns.distplot(qj_dates_all["days"], kde=False, rug=False, hist=True, bins=range(0,1000))
#, bins=[0, 1, 14, 100, 200, 300, 1000])

plt.show(block=True)


'''
for fgrp4 in [3111]:
    print( "----", fgrp4 )
    #                                            for shorter we need a strlen, otherwise we match longer groups
    #print( qjs[ qjs['Function Group'].str.contains('^'+str(fgrp4), na=False) ]["Actual NEW Date"].value_counts() )
    print( qjs[ qjs['Function Group'].str.contains('^'+str(fgrp4), na=False) ] )
'''
