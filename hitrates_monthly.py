#!/usr/bin/env python3
#
#http://patorjk.com/software/taag/#p=display&f=Big&t=DOES%20NOT%20WORK
#
# ------------------------------------------------------------------------------------
#  _____   ____  ______  _____   _   _  ____ _______  __          ______  _____  _  __
# |  __ \ / __ \|  ____|/ ____| | \ | |/ __ \__   __| \ \        / / __ \|  __ \| |/ /
# | |  | | |  | | |__  | (___   |  \| | |  | | | |     \ \  /\  / / |  | | |__) | ' / 
# | |  | | |  | |  __|  \___ \  | . ` | |  | | | |      \ \/  \/ /| |  | |  _  /|  <  
# | |__| | |__| | |____ ____) | | |\  | |__| | | |       \  /\  / | |__| | | \ \| . \ 
# |_____/ \____/|______|_____/  |_| \_|\____/  |_|        \/  \/   \____/|_|  \_\_|\_\
#
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
def plot_qjs(the_df, idx, fgrp, title):
    print( "plot_qjs/the_df" )
    print( the_df.head() )
    print( qjs.head() )
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
            #print( "Skipping (-1) QJ", index )
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
        qj_curr = qj_0
        during_cr = []
        during_cr_m = []
        #print( "----" )
        #print( index, qj_0, qj_1 )
        while qj_curr < qj_1:
            qj_curr_m = qj_curr.strftime('%Y-%m')
            try:
                during_cr.append( the_df.loc[qj_curr_m][idx] )
                during_cr_m.append( qj_curr.replace(day=1) ) # 12 instead of 1 because bars are 24 days, centered
                print( qj_curr_m, the_df.loc[qj_curr_m][idx]  )
            except KeyError:
                pass
            qj_curr += relativedelta(months=1)
        try:
            ave_during_cr = sum(during_cr) / len(during_cr)
        except ZeroDivisionError:
            ave_during_cr = 0.0
        print( during_cr, ave_during_cr )
        # Three months before (maybe use day/time in days, round to months?)
        for d in range(1, 4): # start before, not on
            foo = qj_0 - relativedelta(months=d)
            foo_m = foo.strftime('%Y-%m') # start month
            try:
                #print( "df_loc", foo_m, the_df.loc[foo_m][idx] )
                pre_cr.append( the_df.loc[foo_m][idx] )
                pre_cr_m.append( foo.replace(day=1) ) # 12 instead of 1 because bars are 24 days, centered
                #print( "PRE", index, foo, "/", foo.replace(day=1), "/",
                #       the_df.loc[foo_m][idx] )
            except KeyError:
                pass
        # three months after, skip one
        for d in range(2, 5): #skip one month, so start at next
            foo = qj_1 + relativedelta(months=d)
            foo_m = foo.strftime('%Y-%m') # start month
            try:
                #print( foo_m, the_df.loc[foo_m][idx] )
                post_cr.append(the_df.loc[foo_m][idx])
                post_cr_m.append( foo.replace(day=1) )
                #print( "POST", index, foo, "/", foo.replace(day=1), "/",
                #       the_df.loc[foo_m][idx] )
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
        claim_rates.append( [qj_0, ave_pre_cr, qj_1, ave_post_cr, days] )
        dt_0  = qj_0 - dt_t  # start of days period before QJ open
        dt_1  = qj_1 + dt_t  # end of days period after QJ close
        dt_1s = qj_1 + dt_gap  # start of period affter gap after closing
        dt_1e = dt_1s + dt_t  # end of days period after QJ close
    try:
        print( index, claim_rates )
    except UnboundLocalError:
        pass # happens if no Qjs?
    for i,values in enumerate(claim_rates):
        mean_before = values[1]
        mean_after  = values[3]
        qj_open     = values[0]
        qj_close    = values[2]
        days        = values[4]
        try:
            hitrate = mean_after / mean_before
        except ZeroDivisionError:
            hitrate = 0
        mean_during = abs((mean_after + mean_before)) / 2.0 # JUST FOR TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print( "--->", [ i, fgrp, time_str(qj_open), time_str(qj_close), days, flt(mean_before),
                         flt(mean_during), flt(mean_after), flt(hitrate) ] )
        qjs_info.append( [ i, fgrp, qj_open, qj_close, days, mean_before,
                         mean_during, mean_after, hitrate ] )
        # we know this loops in parallel with the months
    return qjs_info


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
parser.add_argument( '-q', "--plot_qjs", type=str, default=None, help='Which QJs to plot' )
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

fgrp_list = args.fgrp.split(",")


# Precalculate the warrantty volume over all the data so we can use
# it to normalise later on
precalc_months       = 18*12
precalc_warranty_vol = [0] * (precalc_months+24)
precalc_warranty_mon = []
date_lo_dt           = datetime.strptime("2004-01", '%Y-%m')
for m in range( 0, precalc_months ): # ten years
    prod_dt   = date_lo_dt + relativedelta(months = m)
    prod_date = str(prod_dt)[0:7] # YYYY-MM
    precalc_warranty_mon.append( prod_date ) # as string
    quux = foo.asmbvol[ (foo.asmbvol[args.date] == prod_date) ] #select all produced in this month
    chassis_ids = pd.unique(quux["CHASSIS_ID"])
    for warr_m in range(0, 24): # Calculate warranty volume 
        precalc_warranty_vol[m+warr_m] += len(chassis_ids) # add this monthly production volume to the next 24 months
#print( precalc_warranty_mon )
#print( precalc_warranty_vol )
precalc_warranty_vol_df = pd.DataFrame( precalc_warranty_vol[:-24], index=precalc_warranty_mon )
precalc_warranty_vol = dict( zip(precalc_warranty_mon, precalc_warranty_vol) )
#precalc_warranty_vol_df.plot()
#
# ----

all_qjs_cm_info = []
all_qjs_pm_info = []
for fgrp in fgrp_list: 
    qjs, claims, combined, repd_combined = foo.select_fgrp( int(fgrp) )
    qjs = foo.select_qjs_fgrp( fgrp )
    qjs = qjs[ (qjs.qj_open > args.prod_date) ]
    print( "qjs.shape", qjs.shape )
    #
    # Taking all vehicles in a certain prod month, look/plot the claims
    #
    prod_date    = args.prod_date
    base_prod_dt = datetime.strptime(args.prod_date, '%Y-%m')
    filenames    = "" # just for convenient terminal copy/past at end
    plotall = []
    warranty_months  = args.warranty_months
    all_prod_dates   = []
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
        #quux = foo.asmbvol[ (foo.asmbvol['VEH_ASSEMB_MONTH'] == prod_date) ]
        quux = foo.asmbvol[ (foo.asmbvol[args.date] == prod_date) ]
        #print( quux[["CHASSIS_ID", "VEH_ASSEMB_MONTH", "RETAIL_DATE", "RETAIL_MONTH"]].head(4) )
        chassis_ids = pd.unique(quux["CHASSIS_ID"])
        #print( chassis_ids )
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
    #
    print( "all_claims_prodmonth" )
    print( all_claims_prodmonth )
    print( "all_warranty_vol", len(all_warranty_vol), "\n", all_warranty_vol )
    #
    # 3d plot
    colours = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
               "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
               "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f"]
    #          "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
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
        month_idx += 1
    #
    all_padded_df = pd.DataFrame( all_padded, index=months_str )
    #print( all_padded_df )
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
    print( "all_claims_claimmonth_df" )
    print( all_claims_claimmonth_df )
    #
    # Print/calculate QJ info
    #
    all_qi = plot_qjs( all_claims_claimmonth_df, "normalised", fgrp, "claim month" ) #############################
    for x in all_qi:
        all_qjs_cm_info.append( x )
    print( all_qjs_cm_info )
    #
    # Prod month totals
    # We should have one normalised on production volume as well!
    #
    #print( all_claims_prodmonth )
    #print( all_claims_prodmonth_prodnorm )
    all_claims_prodmonth_df = pd.DataFrame( all_claims_prodmonth, index=months_str )
    print( "all_claims_prodmonth_df" )
    print( all_claims_prodmonth_df )
    #
    # Claims on production month
    #
    # QJ
    all_qi = plot_qjs( all_claims_prodmonth_df, "normalised", fgrp, "prod month" ) #############################
    for x in all_qi:
        print( "x", x )
        all_qjs_pm_info.append( x )
    print( all_qjs_pm_info )
    #
all_qjs_cm_info_df = pd.DataFrame( all_qjs_cm_info, columns=["idx", "fgrp4", "qj_open", "qj_close", "days",
                                                       "cr_before", "cr_durin", "cr_after", "hitrate"] )
fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(12,8), sharex=True) #QJ scatter
scatter(all_qjs_cm_info_df, ax4, "claim  month", (0, 1000), (0, 6))
all_qjs_pm_info_df = pd.DataFrame( all_qjs_pm_info, columns=["idx", "fgrp4", "qj_open", "qj_close", "days",
                                                       "cr_before", "cr_durin", "cr_after", "hitrate"] )
fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(12,8), sharex=True) #QJ scatter
scatter(all_qjs_pm_info_df, ax5, "prod month", (0, 1000), (0, 6))
plt.show(block=args.block)

