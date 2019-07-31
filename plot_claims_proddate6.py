#!/usr/bin/env python3
#
# ------------------------------------------------------------------------------------
#  Plots claim rates and overlays QJs, warranty and production volumes.
#    extra options to plot 3D graphs and monthly claims after production month graphs.
#
# Example
#  python3 plot_claims_proddate6.py -f 8752 -p 2011-01 -m84 -3 -i -q all
#
#  python3 plot_claims_proddate6.py -f 8752 -p 2011-01 -b -m0 -i
# ------------------------------------------------------------------------------------
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
import pickle

from read_data import *
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot

pd.set_option('display.width', 120)

mpl_colours = [ "#1f77b4", "#ff7fe0", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
cat20_colours = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
                 "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
                 "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", 
                 "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

plot_size = (10,6) #(12, 8)
plot_dpi = 144
plot_prefix = "TMP" 

def time_str(t):
    return (str(t)[0:10]).replace("-", "")

def flt(f):
    return "{:.2f}".format(f)

def scatter(foo, ax4, title, xlims, ylims):
    # scatter plot on days/hitrate
    # Filter out extreme ones?
    #foo     = foo[ (foo['hitrate'] >= 0) & (foo['hitrate'] < 4) ]
    info    = "hitrates"
    # Also filter length?
    foo     = foo[ (foo['days'] > 0) & (foo['days'] < 3*365) ]
    ##info   += "\ndays: [ " + str(args.days_min) +", "+ str(args.days_max) + " )"
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

# Called from the ax5 and ax6 plots plotting claimsprodmonth and claimsclaimmonth
def plot_qjs(ax, the_df, idx, fgrp, title):
    print( "plot_qjs" )
    print( the_df.head() )
    #
    # Print/calculate QJ info
    #
    qjs_info           = [] #list with all info ?
    claim_rates        = []
    pre_claim_rates_m  = [] #dates (months) used to calculate
    post_claim_rates_m = [] #dates (months) used to calculate
    for index, row in qjs.iterrows():
        if not args.plot_qjs:
            continue
        if args.plot_qjs != "all":
            print( index, args.plot_qjs.split(",") )
            if not str(index) in args.plot_qjs.split(","):
                continue
        qj_0   = row["qj_open"] # real date/time
        qj_1   = row["qj_close"]# real date/time
        if qj_0 == -1 or qj_1 == -1:
            print( "Skipping (-1) QJ", index )
            continue
        days   = qj_1 - qj_0
        days   = int(str(days).split()[0]) # +1 removed, 0 days means "quick fix"
        if  not (0 <= days < 3*365):
            print( "SKIP DAYS LIMIT:", row["qj_id"], "/", index, days )
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
        qj_curr = qj_0
        during_cr = []
        during_cr_m = []
        print( "----" )
        print( index, qj_0, qj_1 )
        while qj_curr < qj_1:
            qj_curr_m = qj_curr.strftime('%Y-%m')
            try:
                during_cr.append( the_df.loc[qj_curr_m][idx] )
                during_cr_m.append( qj_curr.replace(day=1) ) # 12 instead of 1 because bars are 24 days, centered
                print( "CURR", index, qj_curr_m, the_df.loc[qj_curr_m][idx]  )
            except KeyError:
                pass
            qj_curr += relativedelta(months=1)
        try:
            ave_during_cr = sum(during_cr) / len(during_cr)
        except ZeroDivisionError:
            ave_during_cr = 0.0
        print( during_cr, ave_during_cr )
        # Three months before (maybe use day/time in days, round to months?)
        #for d in range(1, 4): # start before, not on
        for d in range(1, args.period_before+1): # start before, not on. 1 is gap (=0), meaning previous month 
            foo = qj_0 - relativedelta(months=d)
            foo_m = foo.strftime('%Y-%m') # start month
            try:
                #print( "df_loc", foo_m, the_df.loc[foo_m][idx] )
                pre_cr.append( the_df.loc[foo_m][idx] )
                pre_cr_m.append( foo.replace(day=1) ) # 12 instead of 1 because bars are 24 days, centered
                print( "PRE", index, foo, "/", foo.replace(day=1), "/", the_df.loc[foo_m][idx] )
            except KeyError:
                pass
        # three months after, skip one
        #for d in range(2, 5): #skip one month, so start at next
        for d in range(args.gap+1, args.period_after+args.gap+1): #skip one month, so start at next   
            foo = qj_1 + relativedelta(months=d)
            foo_m = foo.strftime('%Y-%m') # start month
            try:
                #print( foo_m, the_df.loc[foo_m][idx] )
                post_cr.append(the_df.loc[foo_m][idx])
                post_cr_m.append( foo.replace(day=1) )
                print( "POST", index, foo, "/", foo.replace(day=1), "/", the_df.loc[foo_m][idx] )
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
        print( index, ave_pre_cr, ave_post_cr )
        claim_rates.append( [qj_0, ave_pre_cr, qj_1, ave_post_cr, days, ave_during_cr] )
        pre_claim_rates_m.append( pre_cr_m )
        post_claim_rates_m.append( post_cr_m )
        dt_0  = qj_0 - dt_t  # start of days period before QJ open
        dt_1  = qj_1 + dt_t  # end of days period after QJ close
        dt_1s = qj_1 + dt_gap  # start of period affter gap after closing
        dt_1e = dt_1s + dt_t  # end of days period after QJ close
    try:
        print( index, claim_rates )
    except UnboundLocalError:
        pass # happens if no Qjs?
    for l in pre_claim_rates_m:
        print( [x.strftime('%Y-%m') for x in l] )
    for l in post_claim_rates_m:
        print( [x.strftime('%Y-%m') for x in l] )
    for i,values in enumerate(claim_rates):
        c = colours[ (i % (len(colours)-1)) + 1 ] # if i==0 we get some colour as bars
        mean_before  = values[1]
        mean_after   = values[3]
        mean_during  = values[5]
        qj_open      = values[0]
        qj_close     = values[2]
        days         = values[4]
        hitrate      = mean_after / mean_before
        print( "--->", [ i, fgrp, time_str(qj_open), time_str(qj_close), days, flt(mean_before),
                         flt(mean_during), flt(mean_after), flt(hitrate) ] )
        qjs_info.append( [ i, fgrp, qj_open, qj_close, days, mean_before,
                         mean_during, mean_after, hitrate ] )
        # we know this loops in parallel with the months
        pre_cr_ms  = pre_claim_rates_m[i] 
        post_cr_ms = post_claim_rates_m[i]
        # The claim period itself
        ax.plot( (qj_open, qj_close),
                  (mean_during, mean_during),
                  linewidth=4,
                  color=c)
        # horizontal bar in pre-preriod
        try:
            # we replace the days at (beginning and) end to get full month coverage
            # (and this one is "reversed")
            # months bars are "24 days"
            ax.plot( (pre_cr_ms[0].replace(day=24), pre_cr_ms[-1].replace(day=1)),
                      (mean_before, mean_before),
                      linewidth=4,
                      color=c)
        except IndexError:
            pass
        # vertical line open
        ax.vlines( x=qj_open, #open date
                    ymin=0, ymax=max(mean_before, mean_during),
                    color=c, linestyle=":", linewidth=2)
        # vertical line close
        ax.vlines( x=qj_close, #close date
                    ymin=0, ymax=max(mean_after, mean_during),
                    color=c, linestyle=":", linewidth=2)
        # horizontal bar in post-period
        try:
            ax.plot( (post_cr_ms[0].replace(day=1), post_cr_ms[-1].replace(day=24)),
                      (mean_after, mean_after),
                      linewidth=4,
                      color=c)
        except IndexError:
            pass
    #plot scatter
    qjs_info_df = pd.DataFrame( qjs_info, columns=["idx", "fgrp4", "qj_open", "qj_close", "days",
                                                   "cr_before", "cr_durin", "cr_after", "hitrate"] )
    fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(12,8), sharex=True) #QJ scatter
    scatter( qjs_info_df, ax4, title, (0,2*365), (0, 4) )


'''
2584    5635
3111    5411
2581    3017
2846    2577
3902    2417
3651    2169
2841    1482
5931    1314
2346    1231
'''

'''
lvd_chassis_ids.csv
T_CHASSIS,REP_DATE,VEH_ASSEMB_DATE
B-817917,0,2017-03-09
B-747931,2017-09-07,2015-10-06
...
'''
parser = argparse.ArgumentParser()
parser.add_argument( '-a', "--anon", action='store_true', default=False, help='Anonymous axes' )
parser.add_argument( '-b', "--block", action='store_true', default=False, help='Keep showing the plot until closed' )
parser.add_argument( '-f', '--fgrp', type=str, default="3110,3111", help='Specify FGRP4s')
parser.add_argument( '-p', "--prod_date", type=str, default="2011-01", help='Production month' )
parser.add_argument( '-m', "--months", type=int, default=120, help='Plot this many months' )
parser.add_argument( '-w', "--warranty_months", type=int, default=24, help='Warranty period' )
parser.add_argument( '-d', "--date", type=str, default="VEH_ASSEMB_MONTH", help='Which field to use' ) # RETAIL_MONTH
parser.add_argument( '-y', "--ymax", type=float, default=None, help='Y-axis maximum' )
parser.add_argument( '-x', "--xmax", type=int, default=760, help='X-axis size in days' ) # now warranty_months
parser.add_argument( '-n', "--normalise", action='store_true', default=False, help='Normalise claims' )
parser.add_argument( '-s', "--stagger", action='store_true', default=False, help='Stagger plots on absolute months' )
parser.add_argument( '-l', "--labels", action='store_true', default=False, help='Show labels on bars' )
parser.add_argument( '-3', "--threed", action='store_true', default=False, help='Show 3D plot' )
parser.add_argument( '-i', "--individual_months", action='store_true', default=False, help='Show monthly plots' )
parser.add_argument( '-L', "--lvd_ids", type=str, default=None, help='Use LVD chassis ids only' )
parser.add_argument( '-q', "--plot_qjs", type=str, default=None, help='Which QJs to plot' )
#
parser.add_argument( '-g', "--gap", type=int, default=1, help='Gap after QJ closing month in months' )
parser.add_argument( "--period_after", type=int, default=3, help='Number ofmonths to calculate claim rate on' )
parser.add_argument( "--period_before", type=int, default=3, help='Number ofmonths to calculate claim rate on' )
args = parser.parse_args()

colours = ["#0049E5", "#00C3E1", "#00D906", "#6FD500", "#D2C300", "#2100BF", "#CE4E00", "#CA0022",
           "#C6008F", "#8D00C2", "#00DD80",
           "#6F00E5","#0002E1","#0072DD","#00D9D5","#00D566",
           "#04D200","#6CCE00","#CAC400","#C65D00", "#C20006","#BF0066"]

if os.path.exists( "foo.pickle" ):
    with open("foo.pickle", "rb") as f:
        foo = pickle.load( f )
else:
    foo = DATA(verbose=True)
    foo.read_full()
    with open("foo.pickle", "wb") as f:
        pickle.dump( foo, f )
#foo = DATA(verbose=True)
#foo.read_full()

fgrp_list = args.fgrp.split(",")

lvd_chassis_ids = None
if args.lvd_ids: # If specified, filter on these chassis ids.
    lvd_chassis_ids = pd.read_csv( args.lvd_ids )

# Precalculate the warrantty volume over all the data so we can use
# it to normalise later on. Start in 2004.
date_lo_dt           = datetime.strptime("2004-01", '%Y-%m')
date_now_dt          = datetime.now() # actually, data until 2019-02 or so
rd                   = relativedelta(date_now_dt, date_lo_dt)
precalc_months       = rd.years * 12 + rd.months
precalc_months       = 18*12 + 3
precalc_warranty_vol = [0] * (precalc_months+24) # until the end, plus two years
precalc_warranty_mon = []

for m in range( 0, precalc_months ): # ten years
    prod_dt   = date_lo_dt + relativedelta(months = m)
    prod_date = str(prod_dt)[0:7] # YYYY-MM
    precalc_warranty_mon.append( prod_date ) # as string
    quux = foo.asmbvol[ (foo.asmbvol[args.date] == prod_date) ] #select all produced in this month
    chassis_ids = pd.unique(quux["CHASSIS_ID"])
    if args.lvd_ids: # This could (should) be done earlier
        lvd_uniqs   = pd.unique(lvd_chassis_ids["T_CHASSIS"]) # Reza's file
        chassis_ids = list( set(chassis_ids).intersection(set(lvd_uniqs)) ) # take intersection only
    for warr_m in range(0, 24): # Calculate warranty volume 
        precalc_warranty_vol[m+warr_m] += len(chassis_ids) # add this monthly production volume to the next 24 months
#print( precalc_warranty_mon )
#print( precalc_warranty_vol )
precalc_warranty_vol_df = pd.DataFrame( precalc_warranty_vol[:-24], index=precalc_warranty_mon )
precalc_warranty_vol = dict( zip(precalc_warranty_mon, precalc_warranty_vol) )
#precalc_warranty_vol_df.plot()
#
# ----

for fgrp in fgrp_list: # SHOULD ONLY BE ONE!
    qjs, claims, combined, repd_combined = foo.select_fgrp( int(fgrp) )
    qjs = foo.select_qjs_fgrp( fgrp )
    qjs = qjs[ (qjs.qj_open > args.prod_date) ]
    print( "qjs.shape", qjs.shape )
    #print( combined )
    #
    # Taking all vehicles in a certain prod month, look/plot the claims
    #
    prod_date    = args.prod_date
    base_prod_dt = datetime.strptime(args.prod_date, '%Y-%m')
    filenames    = "" # just for convenient terminal copy/past at end
    plotall = []
    warranty_months = args.warranty_months
    all_prod_dates = []
    all_prod_volumes = [] # list with number produced each month
    all_claims_prodmonth = [] # list with sum of claims on prod month, normalised on warranty volume
    all_claims_prodmonth_prodnorm = [] # list with sum of claims on prod month, normalised on prod volume
    all_warranty_vol = [0] * (args.months+1+args.warranty_months)
    for m in range( 0, args.months+1 ):
        prod_dt       = base_prod_dt + relativedelta(months = m)
        prod_date     = str(prod_dt)[0:7] #YYYY-MM
        prod_date_idx = m
        all_prod_dates.append( prod_dt )
        #
        print( m, prod_date, prod_dt )
        plotall.append( m )
        plotall[m] = [ 0 ] * (warranty_months+1)
    plotall = pd.DataFrame( plotall, index=all_prod_dates )
    #print( plotall )
    #
    foo.asmbvol['RETAIL_MONTH'] = foo.asmbvol['RETAIL_DATE'].dt.strftime('%Y-%m')
    print( "---- foo.asmbvol" )
    print( foo.asmbvol )
    print( "----" )
    #
    for m in range( 0, args.months+1 ):
        prod_dt   = base_prod_dt + relativedelta(months = m)
        prod_date = str(prod_dt)[0:7]#datetime.strptime(prod_dt, '%Y-%m')
        #print( m, prod_date, prod_dt )
        '''
        for x in list(foo.asmbvol.columns):
            if "DATE" in x:
                print( x ) #RETAIL_DATE?
        '''
        #quux = foo.asmbvol[ (foo.asmbvol['VEH_ASSEMB_MONTH'] == prod_date) ]
        quux = foo.asmbvol[ (foo.asmbvol[args.date] == prod_date) ]
        #print( quux[["CHASSIS_ID", "VEH_ASSEMB_MONTH", "RETAIL_DATE", "RETAIL_MONTH"]].head(4) )
        chassis_ids = pd.unique(quux["CHASSIS_ID"])
        #print( chassis_ids )
        if args.lvd_ids: # This could (should) be done earlier
            lvd_uniqs = pd.unique(lvd_chassis_ids["T_CHASSIS"]) # Reza's file
            chassis_ids = list( set(chassis_ids).intersection(set(lvd_uniqs)) ) # take intersection only
            print( chassis_ids )
        #
        #print( chassis_ids, len(chassis_ids) )
        # need a "get claims per chassis-id? join/merge, subselect. These are frgp4 3111 ^^
        #print( claims.head(), list(claims.columns) )
        new_df     = pd.DataFrame( columns=["CHASSIS_ID", "DT", "FGRP4"] )
        new_df_lst = []
        num_chassis_with_claim = 0
        total_claims  = 0 # for this prod month
        file_counter  = 0
        for cis in chassis_ids:
            c_claims = claims[(claims["CHASSIS_ID"]==cis)]
            if not c_claims.empty:
                c_claims.to_csv('c_claims{:02n}.csv'.format(file_counter), index=False, sep=";")
                file_counter += 1
                num_chassis_with_claim += 1 #+= len(c_claims) #one len(..), could be more, but still one chassis!
                total_claims += len(c_claims) # or +=1 if we count returned chassis?
                #print( cis, len(c_claims) ) #c_claims[["CLAIM_REG_DATE", "FGRP4"]].values )
                for i, row in c_claims.iterrows():
                    new_df_lst.append( [row["CHASSIS_ID"], row["CLAIM_REG_DATE"], row["FGRP4"] ] )
        #print( "---- new_df_lst" )
        #print( new_df_lst )
        #print( "----" )
        # ---- new_df_lst
        # [
        #  ['B8R-169334', Timestamp('2015-07-23 00:00:00'), 3111.0],
        #  ['B8R-169326', Timestamp('2015-10-29 00:00:00'), 3111.0], ...
        #
        # Turn it into a dataframe
        new_df = pd.DataFrame( new_df_lst, columns=["CHASSIS_ID", "DATE", "FGRP4"] )
        new_df = new_df.sort_values(by='DATE')
        #print( new_df )
        if new_df.empty:
            print( "NO DATA" )
            #sys.exit(2)
            all_prod_volumes.append( len(chassis_ids) ) 
            all_claims_prodmonth.append( 0 )
            all_claims_prodmonth_prodnorm.append( 0 )
            try:
                all_warranty_vol[m] = int(precalc_warranty_vol[prod_date])
            except KeyError:
                all_warranty_vol[m] = -100
            continue
        #print( new_df.groupby(new_df["DATE"]).count() )
        #print( new_df.groupby( pd.Grouper(key='DATE', freq='D') ).count() ) #best
        #plotdata = new_df.groupby( pd.Grouper(key='DATE', freq='D') ).count()
        ##plotdata = pd.DataFrame( {'count' : new_df.groupby( ["DATE"] ).size() }).reset_index() #works, next better
        plotdata = pd.DataFrame( {'count' : new_df.groupby( pd.Grouper(key='DATE', freq='M') ).size() }).reset_index()
        #                                                                        , freq='D' #fills days zeroes
        #
        plotdata = plotdata.set_index("DATE")
        #print( "---- plotdata" )
        #print( plotdata.head(4) )
        #print( "----" )
        #
        # Add to plotall
        tmp = plotdata.reset_index(level=0) # This makes a 0...warranty_month index
        for idx, e in enumerate(tmp.values):
            nc = e[1]      #num claims
            #print( idx, nc )
            try:
                if args.normalise:
                    plotall.iloc[m, idx] = nc / len(chassis_ids)  #DONT, WE WANT WARRANTY VOLUME
                else:
                    plotall.iloc[m, idx] = nc
            except IndexError: # catches errors if after the warranty_months period (this happens)
                pass
        #print( plotall )
        #
        #
        #print( "len(chassis_ids)", len(chassis_ids) )
        all_prod_volumes.append( len(chassis_ids) ) # Use this to calculate real prod volume later
        all_warranty_vol[m] = int(precalc_warranty_vol[prod_date]) 
        if all_warranty_vol[m] > 0:
            all_claims_prodmonth.append( total_claims / all_warranty_vol[m] * 1000.0 ) # Normalise on total warranty volume
        else:
            all_claims_prodmonth.append( 0 )
        #
        #all_claims_prodmonth_prodnorm.append( total_claims / len(chassis_ids) * 1000.0  ) # Normalise on prod volume
        all_claims_prodmonth_prodnorm.append( num_chassis_with_claim / len(chassis_ids) * 1000.0  ) 
        #
        if args.normalise:
            print( "Normalise over production volume, "+str(len(chassis_ids)) ) # DONT, WE WANT WARRANTY VOLUME
            plotdata["count"] = plotdata["count"] / len(chassis_ids) 
        #
        if args.individual_months: # True to get plot for each production month
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
            #fig.subplots_adjust(bottom=0.3)
            #
            import matplotlib.dates as mdates
            years    = mdates.YearLocator()   # every year
            months   = mdates.MonthLocator()  # every month
            yearsFmt = mdates.DateFormatter('%Y-%m')
            ax.xaxis.set_major_locator(months) #years)
            ax.xaxis.set_major_formatter(yearsFmt)
            ax.xaxis.set_minor_locator(months)
            if not args.ymax:
                ax.set_ylim(0, max(plotdata["count"])+1)
            else:
                ax.set_ylim(0, args.ymax)
            #ax.set_xlim( prod_dt, prod_dt + timedelta(days=args.xmax) ) # 760 days limit
            ax.set_xlim( prod_dt, prod_dt + relativedelta(months = warranty_months) ) # NEW
            #
            days_in_month = [(plotdata.index[j+1]-plotdata.index[j]).days * -0.8 for j in range(len(plotdata.index)-1)] \
                + [30*-0.8]
            rects = ax.bar( plotdata.index,
                            plotdata["count"],    # normalise over total number produced?
                            width=days_in_month, #24, #in "days"
                            align="edge",
                            color=colours[0],
                           #edgecolor="black",
            )
            #
            if args.labels:
                #print( rects )
                for rect in rects:
                    height = rect.get_height()
                    if args.normalise:
                        unnormalised = int(height * len(chassis_ids))
                    else:
                        unnormalised = int(height)
                    print( "height", height, unnormalised )
                    if unnormalised > 0:
                        ax.text( rect.get_x() + rect.get_width()/2., 1.05*height,
                                 '%d' % unnormalised,
                                 ha='center', va='bottom')
            #
            tot = 0
            for rect in rects:
                height = rect.get_height()
                if args.normalise:
                    # does this make sense? Let's plot absolute numbers
                    #unnormalised = int(height * len(chassis_ids))
                    unnormalised = int(height) 
                else:
                    unnormalised = int(height)
                tot += unnormalised
            #print( tot ) # actually, take plotdata.sum()
            #
            #plotdata.plot( ax=ax )
            fig.autofmt_xdate()
            #
            #ax.xaxis.label.set_size(8)
            plt.xticks( fontsize=8 )
            #
            if args.normalise:
                ax.set_xlabel( 'Absolute number of claims FGRP'+str(fgrp) ) # was Normalised, see above
            else:
                ax.set_xlabel( 'Absolute number of claims FGRP'+str(fgrp) )
            if not args.anon:
                ax.set_title( str(len(chassis_ids))+" / "+args.date+"="+prod_date+" / "+str(num_chassis_with_claim) ) #+" / "+str(tot) )
            else:
                ax.set_title( str(len(chassis_ids))+" / "+args.date+"="+prod_date+" / "+str(num_chassis_with_claim) ) 
            if args.date == "RETAIL_MONTH":
                PDFFILE="retailmonth_"+str(prod_date)+"_fgrp"+str(fgrp)+".png"
            else:
                PDFFILE="prodmonth_"+str(prod_date)+"_fgrp"+str(fgrp)+".png" #PDF
            #
            if args.anon:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.get_legend().remove()
            #
            if os.path.exists( PDFFILE ):
                os.remove( PDFFILE )
            fig.savefig(PDFFILE, dpi=300)
            print( "Saved", PDFFILE )
            plt.close(fig)
            filenames = filenames + " " + PDFFILE
    #
    print( "all_warranty_vol", len(all_warranty_vol), "\n", all_warranty_vol )
    #
    # 3d plot
    colours = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
               "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
               "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f"]
    #          "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
    Xs         = []
    Ys         = []
    Zs         = []
    cols       = []
    months     = []
    month_idx  = 0
    months_str = []
    all_padded = []
    for i, row in plotall.iterrows():
        #print( month_idx, i, row )
        months_str.append( str(i)[0:7] )
        vals = row.values
        #
        padded = [0] * month_idx + vals.tolist() + ([0] * (args.months - month_idx))
        #print( i, padded ) # calculate column sums...
        all_padded.append( padded )
        #vals = padded
        #
        months.append( month_idx )
        for mon,y in enumerate(vals):
            if args.stagger:
                Xs.append( mon+month_idx ) # LIKE THIS WE STAGGER; ABSOLUTE MONTH START
            else:
                Xs.append( mon ) 
            #print( y, vals[y] )
            Ys.append( month_idx )
            Zs.append( y )
            cols.append( colours[month_idx % len(colours)] )
        month_idx += 1
    #
    all_padded_df = pd.DataFrame( all_padded, index=months_str )
    #print( all_padded_df )
    CSVFILE="allpadded_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+".csv"
    if os.path.exists( CSVFILE ):
        os.remove( CSVFILE )
    all_padded_df.to_csv(CSVFILE, sep=";")#, float_format='%.4f')
    #
    sums = all_padded_df.sum(axis=0)
    sums = sums[:-args.warranty_months] # otherwise we get two years of emptyness at the end
    #print( sums.values )
    #print( all_warranty_vol[0:args.months+1] )
    sums_normalised = sums.values / all_warranty_vol[0:args.months+1] * 1000.0 
    sums_labels = []
    for m in range(0, len(sums)):
        sums_labels.append( (base_prod_dt + relativedelta(months = m)) )
    print( sums, sums.values, len(sums.values), len(sums_labels), len(sums_normalised) )
    all_claims_claimmonth_df = pd.DataFrame( sums.values, index=months_str )
    all_claims_claimmonth_df["normalised"] = sums_normalised # add this one too
    all_claims_claimmonth_df["warranty_vol"] = all_warranty_vol[0:args.months+1] # add this one too
    #print( all_claims_claimmonth_df )
    CSVFILE="allclaimsclaimmonth_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.csv"
    if os.path.exists( CSVFILE ):
        os.remove( CSVFILE )
    all_claims_claimmonth_df.to_csv(CSVFILE, sep=";")#, float_format='%.4f')
    print( "SAVED", CSVFILE )
    #
    #fig5, ax56 = plt.subplots(nrows=2, ncols=1, figsize=(8,4))
    #ax5 = ax56[0] # if sharing with fig6
    fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
    rects = ax5.bar( sums_labels, #all_padded_df.columns,
                     #sums.values,
                     sums_normalised,
                     width=24, align="edge",
                     #width=-24, align="edge"
    )
    #
    # Print/calculate QJ info
    #
    plot_qjs( ax5, all_claims_claimmonth_df, "normalised", fgrp, "claim month" )
    '''
    claim_rates        = []
    pre_claim_rates_m  = [] #dates (months) used to calculate
    post_claim_rates_m = [] #dates (months) used to calculate
    for index, row in qjs.iterrows():
        if not args.plot_qjs:
            continue
        if args.plot_qjs != "all":
            print( index, args.plot_qjs.split(",") )
            if not str(index) in args.plot_qjs.split(","):
                continue
        qj_0   = row["qj_open"] # real date/time
        qj_1   = row["qj_close"]# real date/time
        if qj_0 == -1 or qj_1 == -1:
            print( "Skipping QJ", index )
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
        # Three months before (maybe use day/time in days, round to months?)
        #
        for d in range(1, 4): # start before, not on
            foo = qj_0 - relativedelta(months=d)
            foo_m = foo.strftime('%Y-%m') # start month
            try:
                #print( foo_m, all_claims_claimmonth_df.loc[foo_m]["normalised"] )
                pre_cr.append( all_claims_claimmonth_df.loc[foo_m]["normalised"] )
                pre_cr_m.append( foo.replace(day=1) ) # 12 instead of 1 because bars are 24 days, centered
                print( "PRE", index, foo, "/", foo.replace(day=1), "/",
                       all_claims_claimmonth_df.loc[foo_m]["normalised"] )
            except KeyError:
                pass
        # three months after, skip one
        for d in range(1, 4): #skip one month, so start at next
            foo = qj_1 + relativedelta(months=d)
            foo_m = foo.strftime('%Y-%m') # start month
            try:
                #print( foo_m, all_claims_claimmonth_df.loc[foo_m]["normalised"] )
                post_cr.append(all_claims_claimmonth_df.loc[foo_m]["normalised"])
                post_cr_m.append( foo.replace(day=1) )
                print( "POST", index, foo, "/", foo.replace(day=1), "/",
                       all_claims_claimmonth_df.loc[foo_m]["normalised"] )
            except KeyError:
                pass
                #print( foo_m )
        #print( qj_0_m, pre_cr, qj_1_m, post_cr )
        try:
            ave_pre_cr = sum(pre_cr)/len(pre_cr)
        except ZeroDivisionError:
            ave_pre_cr = 0
        try:
            ave_post_cr = sum(post_cr)/len(post_cr)
        except ZeroDivisionError:
            ave_post_cr = 0
        claim_rates.append( [qj_0, ave_pre_cr, qj_1, ave_post_cr] )
        pre_claim_rates_m.append( pre_cr_m )
        post_claim_rates_m.append( post_cr_m )
        dt_0  = qj_0 - dt_t  # start of days period before QJ open
        dt_1  = qj_1 + dt_t  # end of days period after QJ close
        dt_1s = qj_1 + dt_gap  # start of period affter gap after closing
        dt_1e = dt_1s + dt_t  # end of days period after QJ close
    print( index, claim_rates )
    for l in pre_claim_rates_m:
        print( [x.strftime('%Y-%m') for x in l] )
    for l in post_claim_rates_m:
        print( [x.strftime('%Y-%m') for x in l] )
    for i,values in enumerate(claim_rates):
        c = colours[ (i % (len(colours)-1)) + 1 ] # if i==0 we get some colour as bars
        mean_before = values[1]
        mean_after  = values[3]
        qj_open     = values[0]
        qj_close    = values[2]
        mean_during = abs((mean_after + mean_before)) / 2.0 # JUST FOR TESTING
        # we know this loops in parallel with the months
        pre_cr_ms  = pre_claim_rates_m[i] 
        post_cr_ms = post_claim_rates_m[i]
        # The claim period itself
        ax5.plot( (qj_open, qj_close),
                  (mean_during, mean_during),
                  linewidth=4,
                  color=c)
        # horizontal bar in pre-preriod
        try:
            ax5.plot( (pre_cr_ms[0], pre_cr_ms[-1]),
                      (mean_before, mean_before),
                      linewidth=4,
                      color=c)
        except IndexError:
            pass
        # vertical line open
        ax5.vlines( x=qj_open, #open date
                    ymin=0, ymax=max(mean_before, mean_during),
                    color=c, linestyle=":", linewidth=2)
        # vertical line close
        ax5.vlines( x=qj_close, #close date
                    ymin=0, ymax=max(mean_after, mean_during),
                    color=c, linestyle=":", linewidth=2)
        # horizontal bar in post-period
        try:
            ax5.plot( (post_cr_ms[0], post_cr_ms[-1]),
                      (mean_after, mean_after),
                      linewidth=4,
                      color=c)
        except IndexError:
            pass
    '''
    #
    from matplotlib.dates import MonthLocator, YearLocator, DateFormatter
    ax5.xaxis.set_minor_locator(MonthLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]))
    ax5.xaxis.set_major_locator(YearLocator())
    ax5.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig5.autofmt_xdate()
    if not args.anon:
        ax5.set_title( "Claims per claim month, FGRP"+str(fgrp) )
    else:
        ax5.set_title( "Claims per claim month" )
    ax5.set_xlabel( "Normalised on warranty volume" )
    ax5.set_ylabel( "per 1000 vehicles" )
    if args.date == "RETAIL_MONTH":
        PNGFILE="claimsclaimmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    else:
        PNGFILE="claimsclaimmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    if args.plot_qjs:
        qjs_str = "q"+str("-".join(args.plot_qjs.split(",")))
        PNGFILE="claimsclaimmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m_"+qjs_str+".png"
    #
    if args.anon:
        ax5.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)
        #ax5.get_legend().remove()
    #
    if os.path.exists( PNGFILE ):
        os.remove( PNGFILE )
    fig5.savefig(PNGFILE, dpi=300)
    print( "Saved", PNGFILE )
    #
    #print( len(Xs), Xs )
    #print( len(Ys), Ys )
    #print( len(Zs), Zs )
    #
    if args.threed: # True for 3D plot
        from matplotlib import rcParams
        rcParams['axes.labelpad'] = 20
        rcParams['font.size'] = 7
        #rcParams.update({'font.size': 22})
        from mpl_toolkits.mplot3d import Axes3D
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8), subplot_kw={'projection': "3d"} )
        #fig = plt.figure()
        #ax  = fig.gca(projection = '3d')
        ax.bar3d(Xs,
                 Ys,
                 0,
                 #0.8,1.2/(args.months+1),Zs,
                 0.8, 0.1, Zs,
                 #dx, dy, dz,
                 color = cols, alpha=0.5, shade=True)
        ticksy = np.arange(0.5, args.months+1, 1)
        #ticksy = Ys
        #print( ticksy )
        plt.yticks(ticksy, months_str)
        ax.view_init(elev=44., azim=-124) #48, -146
        ax.dist=12
        if not args.anon:
            ax.set_xlabel('Plus months', fontsize=12 )
            ax.set_ylabel('Production month', fontsize=12 )
        #
        if args.anon:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.get_zaxis().set_visible(False)
                ax.set_xlabel('Months after production', fontsize=12 )
                ax.set_ylabel('Production month', fontsize=12 )
                labels = [item.get_text() for item in ax.get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax.set_xticklabels(empty_string_labels)
                labels = [item.get_text() for item in ax.get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                ax.set_yticklabels(empty_string_labels)
                labels = [item.get_text() for item in ax.get_zticklabels()]
                empty_string_labels = ['']*len(labels)
                ax.set_zticklabels(empty_string_labels)
                #ax.get_legend().remove()
        #
        if args.date == "RETAIL_MONTH":
            PDFFILE="3dretailmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+".png" #PDF
        else:
            PDFFILE="3dprodmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+".png" #PDF
        if os.path.exists( PDFFILE ):
            os.remove( PDFFILE )
        fig.savefig(PDFFILE, dpi=300)
        print( "Saved3D", PDFFILE )
    #
    plotall["prodvol"] = all_prod_volumes
    plotall["allwarrantyvol"] = all_warranty_vol[0:args.months+1]
    print( len(all_warranty_vol) )
    #print( plotall )
    #
    if args.date == "RETAIL_MONTH":
        CSVFILE="3dretailmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.csv"
    else:
        CSVFILE="3dprodmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.csv"
    if os.path.exists( CSVFILE ):
        os.remove( CSVFILE )
    plotall.to_csv(CSVFILE, sep=";")#, float_format='%.4f')
    print( "CSVFILE", CSVFILE )
    print( filenames )
    #
    # Line plot
    fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(12,4) )
    plotall.iloc[:,0:-2].T.plot( ax=ax4, legend=None, c="k", alpha=0.3 ) # all rows, all but last 2 columns (prodval, warr vol)
    ax4.set_title( "Claims per $x$-months after production" )
    ax4.set_xlabel( "Absolute numbers" )
    ax4.set_ylabel( "number of claims" )
    #
    if args.anon:
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        #ax4.get_legend().remove()
    #
    # Prod month totals
    # We should have one normalised on production volume as well!
    #
    #print( all_claims_prodmonth )
    #print( all_claims_prodmonth_prodnorm )
    all_claims_prodmonth_df = pd.DataFrame( all_claims_prodmonth, index=months_str )
    #print( all_claims_prodmonth_df )
    CSVFILE="allclaimsprodmonth_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.csv"
    if os.path.exists( CSVFILE ):
        os.remove( CSVFILE )
    all_claims_prodmonth_df.to_csv(CSVFILE, sep=";")#, float_format='%.4f')
    print( "SAVED", CSVFILE )
    #
    # Claims on production month
    #
    fig6, ax6 = plt.subplots(nrows=1, ncols=1, figsize=(8,4) )
    #ax6 = ax56[1]
    #all_claims_prodmonth_df.bar( ax=ax6 ) # all rows, all but last column
    date_labels = []
    for m in range(0, args.months+1):
        #print( base_prod_dt + relativedelta(months = m) )
        date_labels.append( (base_prod_dt + relativedelta(months = m)) )
    print( len(all_claims_prodmonth), len(months_str), len(date_labels) )
    #       ax5 is possible to combine, but they overwrite each other
    rects = ax6.bar( date_labels,
                     all_claims_prodmonth,
                     width=24, align="edge",
    )
    #
    # QJ
    plot_qjs(ax6, all_claims_prodmonth_df, 0, fgrp, "prod month")
    #
    #ax = plt.gca()
    ax6.xaxis.set_minor_locator(MonthLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]))
    #ax.xaxis.set_minor_formatter(DateFormatter('%b'))
    ax6.xaxis.set_major_locator(YearLocator())
    ax6.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig6.autofmt_xdate()
    #
    if not args.anon:
        ax6.set_title( "Claims per production month, FGRP"+str(fgrp) )
    else:
        ax6.set_title( "Claims per production month" )
    ax6.set_xlabel( "Normalised on warranty volume" )
    ax6.set_ylabel( "per 1000 vehicles" )
    if args.date == "RETAIL_MONTH":
        PNGFILE="claimsretailmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    else:
        PNGFILE="claimsprodmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    if args.plot_qjs:
        qjs_str = "q"+str("-".join(args.plot_qjs.split(",")))
        PNGFILE="claimsprodmonths_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m_"+qjs_str+"_g"+str(args.gap)+".png"
    #
    if args.anon:
        ax6.get_xaxis().set_visible(False)
        ax6.get_yaxis().set_visible(False)
        #ax6.get_legend().remove()
    if os.path.exists( PNGFILE ):
        os.remove( PNGFILE )
    fig6.savefig(PNGFILE, dpi=300)
    print( "Saved", PNGFILE )
    #
    all_claims_prodmonth_df = pd.DataFrame( all_claims_prodmonth, index=months_str )
    #print( all_claims_prodmonth_df )
    CSVFILE="allclaimsprodmonth_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.csv"
    if os.path.exists( CSVFILE ):
        os.remove( CSVFILE )
    all_claims_prodmonth_df.to_csv(CSVFILE, sep=";")#, float_format='%.4f')
    print( "SAVED", CSVFILE )
    #
    # Normalised on prod volume
    fig7, ax7 = plt.subplots(nrows=1, ncols=1, figsize=(8,4) )
    rects = ax7.bar( date_labels,
                     all_claims_prodmonth_prodnorm,
                     width=24, #days
                     color=colours[2],
                     align="edge",
    )
    ax7.xaxis.set_minor_locator(MonthLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]))
    ax7.xaxis.set_major_locator(YearLocator())
    ax7.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig7.autofmt_xdate()
    #
    if not args.anon:
        ax7.set_title( "Chassis with claims per production month, FGRP"+str(fgrp) )
    else:
        ax7.set_title( "Chassis with claims per production month" )
    ax7.set_xlabel( "Normalised on production volume" )
    ax7.set_ylabel( "per 1000 vehicles" )
    if args.date == "RETAIL_MONTH":
        PNGFILE="claimsretailmonthsprodvol_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    else:
        PNGFILE="claimsprodmonthsprodvol_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    #
    if args.anon:
        ax7.get_xaxis().set_visible(False)
        ax7.get_yaxis().set_visible(False)
        #ax7.get_legend().remove()
    if os.path.exists( PNGFILE ):
        os.remove( PNGFILE )
    fig7.savefig(PNGFILE, dpi=300)
    print( "Saved", PNGFILE )
    #
    fig8, ax8 = plt.subplots(nrows=1, ncols=1, figsize=(8,4) )
    rects = ax8.bar( date_labels,
                     all_warranty_vol[0:args.months+1],
                     width=24, #days
                     color=colours[4],
                     align="edge",
    )
    ax8.xaxis.set_minor_locator(MonthLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]))
    ax8.xaxis.set_major_locator(YearLocator())
    ax8.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig8.autofmt_xdate()
    #
    ax8.set_title( "Warranty volume" )
    if args.date == "RETAIL_MONTH":
        PNGFILE="warrantyvolume_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    else:
        PNGFILE="warrantyvolume_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    #
    if args.anon:
        ax8.get_xaxis().set_visible(False)
        ax8.get_yaxis().set_visible(False)
        #ax8.get_legend().remove()
    #
    if os.path.exists( PNGFILE ):
        os.remove( PNGFILE )
    fig8.savefig(PNGFILE, dpi=300)
    print( "Saved", PNGFILE )
    #
    fig9, ax9 = plt.subplots(nrows=1, ncols=1, figsize=(8,4) )
    rects = ax9.bar( date_labels,
                     all_prod_volumes[0:args.months+1],
                     width=24, #days
                     color=colours[4],
                     align="edge",
    )
    ax9.xaxis.set_minor_locator(MonthLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]))
    ax9.xaxis.set_major_locator(YearLocator())
    ax9.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig9.autofmt_xdate()
    #
    ax9.set_title( "Production volume" )
    if args.date == "RETAIL_MONTH":
        PNGFILE="productionvolume_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    else:
        PNGFILE="productionvolume_"+str(args.prod_date)+"_fgrp"+str(fgrp)+"_"+str(args.months)+"m.png"
    #
    if args.anon:
        ax9.get_xaxis().set_visible(False)
        ax9.get_yaxis().set_visible(False)
        #ax9.get_legend().remove()
    #
    if os.path.exists( PNGFILE ):
        os.remove( PNGFILE )
    fig9.savefig(PNGFILE, dpi=300)
    print( "Saved", PNGFILE )
#
plt.show(block=args.block)

