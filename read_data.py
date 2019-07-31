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
#sns.set(color_codes=True)

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

# Filenames for the data files
# claims_excel_filename = "Battery_claims_171016.xlsx" 
# claims_excel_filename = "Bat171016Sub.xlsx"
#claims_excel_filename   = "ALL_CLAIMS_DEU_20190208.xlsx" ###############"All claims_180111.xlsx"
#qj_excel_filename       = "QJ_MR_or_END_Item_List_gtt.xlsx"
#products_excel_filename = "products_deu_171016.xlsx"
#warrvol_excel_filename  = "Nr Vehicles under 2 years warranty per rep mon.xlsx"

# -----------------------------------------------------------------------------
# Classes & Funxions
# -----------------------------------------------------------------------------

def get_fg(x):
    try:
        return int(x.split('-')[0])
    except ValueError:
        # Some have "Unknown" in field
        #print( "ERROR", x )
        return 0
    
'''
Holds data
'''
class DATA():
    def __init__(self, verbose=False):
        # Data sets
        self.claims_excel_filename   = "20190320_All_Claims_warranty.xlsx"
        #                              "All claims_180111.xlsx" #"20190320_All_Claims_warranty.xlsx"
        #"ALL_CLAIMS_DEU_20190225.xlsx" #"All claims_180111.xlsx" ##"ALL_CLAIMS_DEU_20190225.xlsx"
                                      #"ALL_CLAIMS_DEU_20190208.xlsx" #########"All claims_180111.xlsx"
        self.qj_excel_filename       = "QJ_MR_or_END_Item_List_201903_gtt.xlsx" #"QJ_MR_or_END_Item_List_gtt.xlsx"
        self.products_excel_filename = "Product_DEU_20190327.xlsx" #"products_deu_171016.xlsx"
        self.warrvol_excel_filename  = "Time of Event 1_10_04_19.xlsx" #"Nr Vehicles under 2 years warranty per rep mon.xlsx"
        #
        self.qjdate_open  = "Actual NEW Date" # Opening of QJ
        self.qjdate_close = "last MR" # Closing of QJ ?
        # These are created by load_qjs
        self.qjs          = None
        self.qj_dates_all = None
        self.qj_fgrps     = None
        # Created by read_claims
        self.df           = None #### THIS IS NOW self.claims
        self.claims       = None
        # Created by ead_products
        self.asmbvol      = None
        # create_total_volume()
        self.totvol       = None
        self.combined     = None
        # calc_wm_claims()
        self.claim_df     = None
        #
        self.reread = False
        self.days   = 90
        #
        self.verbose = verbose
    def read_full(self):
        '''
        Read the excel data without making FGRP4 and other subselections.
        Only subselection made is German Volvo trucks.
        The filenames are hardcoded.
        '''
        # All the QJ information
        # ----------------------
        sep( "QJ Information" )
        print( "Reading", self.qj_excel_filename )
        if os.path.exists( self.qj_excel_filename+".pickle" ) and not self.reread:
            self.qjs = pd.read_pickle( self.qj_excel_filename+".pickle" )
        else:
            self.qjs = pd.read_excel( self.qj_excel_filename, skiprows=3, usecols='B:AR' )
            self.qjs.to_pickle( self.qj_excel_filename+".pickle" )
        if self.verbose:
            print( self.qjs.head() )
            print( "  qjs.shape", self.qjs.shape )

        qj_frames =  [ pd.read_excel( fn, skiprows=3, usecols='B:AR' ) for fn in ["QJ_MR_or_END_Item_List_201903_gtt.xlsx", "QJ_MR_or_END_Item_List_gtt.xlsx"] ]
        self.qjs = pd.concat( qj_frames, ignore_index=True, axis=0 )
        print( self.qjs["Function Group"].head() ) #columns )
        #self.qjs.dropna()
        #print( self.qjs["Function Group"].value_counts )
        #sys.exit(1)
        
        # Select Volvo trucks
        if self.verbose:
            print( "Selecting Volvo trucks (Product Type = F...)" )
        #self.qjs = self.qjs[ (self.qjs['Product Type'].str.contains('^F', na=False)) ]
        if self.verbose:
            print( "  qjs.shape", self.qjs.shape )

        # Global QJ info over all QJs. We plot this later, at the end with
        # the rest of the plots.
        # There must be QJs started now, but no end yet.
        '''
        self.qj_dates_all = pd.DataFrame(columns=["qj_open", "qj_close", "days", "f_effect", "m_effect", "m_open"])
        qj_index_all = []
        if self.verbose:
            print( "Selecting QJ opening dates later than 2000-01-01" )
        qj_open_all  = self.qjs[ (self.qjs[self.qjdate_open] >= '2000-01-01') ] # not really necessary
        if self.verbose:
            print( "  qj_open_all.shape", qj_open_all.shape )
            print( "Assembling QJs" ) # Slow, if not for one FGRP4
        for i, row in qj_open_all.iterrows():
            od = row[self.qjdate_open]
            cd = row[self.qjdate_close]
            fe = row["Forecasted effectiveness %"]
            me = row["Effectiveness %"] # "measured" effectiveness
            ds = int(str(cd - od).split()[0]) + 1 #length of QJ
            hash = str(od)+str(cd)
            if hash not in qj_index_all: # assume only one opened on a certain date
                self.qj_dates_all = self.qj_dates_all.append( {'qj_open': od, 'qj_close': cd, 'days': ds,
                                                               "f_effect":fe, "m_effect":me,
                                                               'm_open':str(od)[0:7]}, ignore_index=True )
                qj_index_all.append( od )
        self.qj_dates_all = self.qj_dates_all.fillna(-1) # -1s for NaNs
        print( "qj_dates_all shape, head and tail" )
        print( self.qj_dates.shape )
        print( self.qj_dates_all.head() )
        print( self.qj_dates_all.tail() )
        '''
        #print( qj_dates_all.groupby(['m_open']).sum().reset_index() ) # .groupby('id').mean()
        #print( qj_dates_all.groupby('m_open', as_index=False)['days'].mean() )

        # Save all relevant fgrps in a data sctructure, so we can calculate
        # overall numbers, we scan for fgrp4, fgrp3 and fgrp2
        #                                       #longest first, otherwise missed
        # The Function group needs some cleaning
        #
        self.qjs['Function Group'] = self.qjs.loc[:,'Function Group'].map( lambda x: get_fg(x) )            
        #self.qj_fgrps = self.qjs['Function Group'].str.extract('(?P<fgrp4>\d{4})|(?P<fgrp3>\d{3})|(?P<fgrp2>\d{2})',
        #                                                       expand=True).fillna(0).astype(int)
        self.qj_fgrps = self.qjs['Function Group']
        if self.verbose:
            print( self.qj_fgrps.head() )
            #print( qj_fgrps['fgrp4'].drop_duplicates().tolist() )
            print( "  qj_fgrps.shape", self.qj_fgrps.shape )
        #print( self.qj_fgrps.drop_duplicates().tolist() )
        
        # All FGRP4s
        # ----------
        #self.fgrp4_list = [ x for x in sorted(self.qj_fgrps['fgrp4'].unique()) if x > 999 ]
        self.fgrp4_list = [ int(x) for x in sorted(self.qj_fgrps.unique()) if int(x) > 999 ]
        '''
        for fgrp4 in fgrp4_list:
            print( "----", fgrp4 )
            #                                            for shorter we need a strlen, otherwise we match longer groups
            print( qj[ qj['Function Group'].str.contains('^'+str(fgrp4), na=False) ]["Actual NEW Date"].value_counts() )
            if fgrp4 == 3111:
                break
        '''

        # Claims per day/time period
        # --------------------------
        sep( "Reading claims data" )
        print( "Reading", self.claims_excel_filename )
        if os.path.exists( self.claims_excel_filename+".pickle" ) and not self.reread:
            self.claims = pd.read_pickle( self.claims_excel_filename+".pickle" )
        else:
            self.claims = pd.read_excel( self.claims_excel_filename ) # TAKES A LONG TIME (2 MINUTES)
            self.claims.to_pickle( self.claims_excel_filename+".pickle" )
        if self.verbose:
            print( self.claims.head() )
            print( "  df.shape", self.claims.shape )

        #print( self.claims["CLAIM_CATEGORY"].value_counts() )
        
        # Trucks, volume per month (assembly is not sold?), cumulative
        # ------------------------------------------------------------
        sep( "Reading product information" )
        print( "Reading", self.products_excel_filename )
        if os.path.exists( self.products_excel_filename+".pickle" ) and not self.reread:
            self.asmbvol = pd.read_pickle( self.products_excel_filename+".pickle" )
        else:
            self.asmbvol = pd.read_excel( self.products_excel_filename )
            self.asmbvol = self.asmbvol[ (self.asmbvol['VEH_TYPE'].str.contains('^F', na=False)) ]
            self.asmbvol.to_pickle( self.products_excel_filename+".pickle" )
        if self.verbose:
            print( self.asmbvol.head() )
            print( "  asmbvol.shape", self.asmbvol.shape )
        '''
        We also only have the "F..." vehicle type like in data loaded
        above:
        asmbvol['VEH_TYPE'].value_counts()
        FH13  21917    FM11  1643    FM13  1232    FL8    816
        FE8     486    FMX1   343    FH16   330    FL5    309
        '''
        if self.verbose:
            print( "Aggregate on VEH_ASSEMB_DATE to create total volume" )
        # subselect on something here? fgrp, model? No, later
        # cumsums contains the number of trucks assembled
        # on a given date.
        cumsums = self.asmbvol.groupby('VEH_ASSEMB_DATE').size().cumsum() # VEH_AGE_CLAIM
        '''
        In [4]: cumsums
        Out[4]: 
        VEH_ASSEMB_DATE
        2012-11-28       18
        2012-12-03       19
        2012-12-05       22
        2012-12-06       23
        '''
        # Create a series, and a new dataframe
        VEHPROD0 = cumsums.index[0]  # date of first
        VEHPROD1 = cumsums.index[-1] # date of last
        dtr      = pd.date_range( start=VEHPROD0, end=VEHPROD1, freq='D')
        self.totvol = self.create_vol_warr( cumsums, dtr, "TOTVOL", "WARRANTYTOTVOL" ) # new dataframe
        # NB this is still on all days
        if self.verbose:
            print( "totvol head and tail:" )
            print( self.totvol.head(4) )
            print( self.totvol.tail(4) )

        # Join claims and vehicles on PRODUCT_ID, so we can get the
        # real production date of the vehicle
        sep( "Merging data" )
        if os.path.exists( "data_merged.pickle" ) and not self.reread:
            self.claims = pd.read_pickle( "data_merged.pickle" )
        else:
            self.claims = pd.merge( left=self.claims, right=self.asmbvol,
                                    left_on='PRODUCT_ID', right_on='PRODUCT_ID' )
            self.claims.to_pickle( "data_merged.pickle" )
        if self.verbose:
            print( self.claims.head() )
            print( "  claims.shape", self.claims.shape )
            print( self.claims.head() )
            print( self.claims.tail() )

        print( self.claims.shape )

        self.claims["FGRP1"] = pd.to_numeric( self.claims["FGRP1"], errors='coerce')
        self.claims["FGRP2"] = pd.to_numeric( self.claims["FGRP2"], errors='coerce')
        self.claims["FGRP3"] = pd.to_numeric( self.claims["FGRP3"], errors='coerce')
        self.claims["FGRP4"] = pd.to_numeric( self.claims["FGRP4"], errors='coerce')

        #print( self.claims["FGRP4"].value_counts() )
        #sys.exit(1)
        
        # (Here we could subselect on vehicle type, like FH8 ?)
        
        # --------------------------------------------------------------------------
        # At this point, we have self.claims containing claims and production dates,
        # and self.qjs containing all the QJs, and self.qj_dates_all containing
        # the opening and closing dates of the QJs.
        # We also have self.totvol containing total population vol/warranty vol
        # --------------------------------------------------------------------------

        # Once more for weekly, monthly (fillna needs NaNs to work)
        #
        totvol_w = self.totvol.resample('W').sum()
        totvol_w = totvol_w.replace(0, np.nan).fillna( method='ffill' )
        if self.verbose:
            print( "totvol_w head and tail" )
            print( totvol_w.head() )
            print( totvol_w.tail() )
        totvol_m = self.totvol.resample('M').sum()
        totvol_m = totvol_m.replace(0, np.nan).fillna( method='ffill' )
        if self.verbose:
            print( "totvol_m head and tail" )
            print( totvol_m.head() )
            print( totvol_m.tail() )

        # MERGE totvol_w AND totvol_m SOMEWHERE?
        #self.totvol = pd.merge( self.totvol, totvol_w, left_index=True, right_index=True )
        #self.totvol = pd.merge( self.totvol, totvol_m, left_index=True, right_index=True )
        #print( "totvol head and tail:" )
        #print( self.totvol.head(4) )
        #print( self.totvol.tail(4) )
        #sys.exit(9)
                
        # Read monthly volume
        # -------------------
        # NOTE; THIS IS NOT USED
        # We use our cacluated volume, because that one we can potentially subselect
        # on smaller populations.
        sep( "Reading vehicles under warranty information" )
        print( "Reading", self.warrvol_excel_filename )
        self.monvol = pd.read_excel( self.warrvol_excel_filename, sheet_name="Table" )
        if self.verbose:
            print( self.monvol.head() )

        # Like above, create a timeseries and dataframe with the
        # warranty volume
        VEHPROD0   = self.monvol['Repair Month'].iloc[0]
        VEHPROD1   = self.monvol['Repair Month'].iloc[-1]
        monvol_dtr = pd.date_range( start=VEHPROD0, end=VEHPROD1, freq='D' )
        self.monvol.set_index( 'Repair Month', inplace=True )
        self.monvol = self.monvol.reindex( monvol_dtr )
        self.monvol = self.monvol.fillna( method='ffill' )
        if self.verbose:
            print( "monvol head and tail" )
            print( self.monvol.head() )
            print( self.monvol.tail() )
            print( "monvol.shape", self.monvol.shape )
    def select_fgrp(self, fgrp):
        if self.verbose:
            print( "Select function group", fgrp )
        if fgrp == -1:
            qjs    = self.qjs
            claims = self.claims
        else:
            qjs    = self.qjs[ (self.qjs['Function Group'] == fgrp) ]
            if 0 <= fgrp < 10:
                claims = self.claims[ (self.claims['FGRP1'] == fgrp) ]
            elif 10 <= fgrp < 100:
                claims = self.claims[ (self.claims['FGRP2'] == fgrp) ]
            elif 100 <= fgrp < 1000:
                claims = self.claims[ (self.claims['FGRP3'] == fgrp) ]
            elif 1000 <= fgrp < 10000:
                claims = self.claims[ (self.claims['FGRP4'] == fgrp) ]
            else:
                print( "ERROR; unknown fgrp specified", fgrp )
                sys.exit(2)
        #print( self.claims['FGRP4'] )
        if self.verbose:
            print( "  FGRP claims.shape", claims.shape )
        if claims.empty:
            if self.verbose:
                print( "  Empty fgrp", fgrp )
            return qjs, claims, pd.DataFrame(), pd.DataFrame()
        
        #now we want to merge the number of claims into this table as well
        #so we get dates, subvol, subwarrantyvol, vol, warrantyvol, claims
        if self.verbose:
            print( "Claim counts for FGRP", fgrp )
        counts    = claims.groupby('VEH_ASSEMB_DATE').size()
        counts_df = counts.to_frame()
        counts_df.columns = ['CLAIMCOUNT']
        repd_counts    = claims.groupby('REP_DATE').size()
        repd_counts_df = repd_counts.to_frame()
        repd_counts_df.columns = ['CLAIMCOUNT']  #NB same columns names, but different dataframe
        if self.verbose:
            print( "counts_df head and tail" )
            print( counts_df.head() )
            print( counts_df.tail() )
            print( "repd_counts_df head and tail" )
            print( repd_counts_df.head() )
            print( repd_counts_df.tail() )

        # merge with volumes on date
        # with how="outer" we get all dates, even when there are no subpop claims
        if self.verbose:
            print( "Merge volumes and counts_df" )
        combined = pd.merge( left=self.totvol, right=counts_df,
                             left_index=True, right_index=True) 
        repd_combined = pd.merge( left=self.totvol, right=repd_counts_df,
                                  left_index=True, right_index=True)
        if combined.empty or repd_combined.empty:
            return qjs, claims, pd.DataFrame(), pd.DataFrame()
        if self.verbose:
            print( "combined head and tail" )
            print( combined.head(4) )
            print( combined.tail(4) )
            print( "repd_combined head and tail" )
            print( repd_combined.head(4) )
            print( repd_combined.tail(4) )

        # Normalise
        # (We are normalising with assembly date, which is the one we are plotting)
        #
        # PJB TODO: because this is fgrp, we can select the relevant QJ, use
        # the volume in column X for normalisation, after Skype meeting 2018-07-10.
        # Of course, they are different per QJ, so in that case, we can only print
        # one Qj, or calculate the hitrate for the particular QJ.
        #
        if self.verbose:
            print( "Normalise" ) #still total number of claims, not subpopulation claims
        combined['TOTNORMALISED'] = combined.apply(
            lambda g: int(g['CLAIMCOUNT']) / int(g['WARRANTYTOTVOL']) * 1000.0,
            axis=1 )
        repd_combined['TOTNORMALISED'] = repd_combined.apply(
            lambda g: int(g['CLAIMCOUNT']) / int(g['WARRANTYTOTVOL']) * 1000.0,
            axis=1 )
        if self.verbose:
            print( "combined head and tail" )
            print( combined.head() )
            print( combined.tail() )
            print( "repd_combined head and tail" )
            print( repd_combined.head() )
            print( repd_combined.tail() )

        return qjs, claims, combined, repd_combined
    def create_vol_warr(self, cumsums, idx, lbl_total, lbl_warranty):
        '''
        Create the volume of truck under 2 year warranty from
        truck volumes.
        '''
        if self.verbose:
            print( "_Create tmp dataframe vol with column", lbl_total )
        vol = pd.DataFrame(index=idx, columns=[lbl_total])
        vol.sort_index(inplace=True)
        if self.verbose:
            print( "_Fill totvol" )
        for i, row in cumsums.iteritems():
            vol.loc[i] = row
        if self.verbose:
            print( "_Forward fill missing days" )
        vol = vol.fillna(method='ffill') # forward fill
        if self.verbose:
            print( "_Change NaNs to 0" )
        vol = vol.fillna(0) # zeros for NaNs
        vol.sort_index(inplace=True)
        # Make a new column with "under warranty" (= 2 years?) by subtracting
        # the volume from two years ago from the total volume. We have all the
        # days in the index, so we can loop and assume all needed data is present.
        if self.verbose:
            print( "_Add", lbl_warranty )
        vol[lbl_warranty] = 0 # Add new "empty" column
        for index, row in vol.iterrows():
            twoyears = index - pd.offsets.DateOffset(years=2) # subtract two years from date (should start +2 years)
            #logging.debug( "warranty "+str(index)+" "+str(row)+" "+str(twoyears) )
            if twoyears in vol.index: 
                vol.loc[index, lbl_warranty] = row[lbl_total] - vol.loc[twoyears, lbl_total]
            else:
                vol.loc[index, lbl_warranty] = row[lbl_total]
        return vol
    # Return QJs for fgrp. The data returned is a subset of the self.qjs data.
    #
    # PJB TODO Maybe we can do the "new" normalisation here?
    #
    def select_qjs_fgrp(self, fgrp):
        # TODO: There is a VMT column as well we can use
        fgrp = int(fgrp)
        if self.verbose:
            print( "Selecting function group", fgrp )
        if fgrp == -1:
            qj = self.qjs
        else:
            qj = self.qjs[ (self.qjs['Function Group'] == fgrp) ]
        if self.verbose:
            print( "  qj.shape", qj.shape )

        # Opening (no claims from before 2013-01-01, no use to look at earlier QJs)
        if self.verbose:
            print( "Selecting QJ opening dates later than 2010-01-01" )
        qjdate_open  = "Actual NEW Date" # Opening of QJ
        qjdate_close = "last MR" # Closing of QJ ?
        #qjdate_close = "Actual MR Month" # Closing of QJ
        qj_open = qj[ (qj[qjdate_open] >= '2010-01-01') ]
        if self.verbose:
            print( "  qj_open.shape", qj_open.shape )

        # Create a datafame with unique opening/closing dates
        # index should prolly include closing date just in case
        if self.verbose:
            print( "Select open and closing dates" )
        qj_dates = pd.DataFrame(columns=["qj_id", "qj_open", "qj_close", "qj_amrm", "f_effect", "m_effect", "fg",
                                         "ntotval"])
        qj_index = []
        for i, row in qj_open.iterrows():
            qi = row["QJ #"]
            od = row[qjdate_open]
            cd = row[qjdate_close]
            am = row["Actual MR Month"]
            fe = row["Forecasted effectiveness %"]
            me = row["Effectiveness %"] # "measured" effectiveness
            fg = int(row["Function Group"]) # "8743-compressor and mounting"
            nt = row["N Tot Value (QJ)"]
            #print( qi, od, cd, am, fe, me, nt )
            hash = str(qi)+str(od)+str(cd)+str(fg)
            if hash not in qj_index: # and fg >= 1000: # assume only one opened/closed on certain dates
                qj_dates = qj_dates.append( {'qj_id':qi, 'qj_open': od, 'qj_close': cd, "qj_amrm":am,
                                             "f_effect":fe, "m_effect":me, "fg":fg,
                                             "ntotval":nt}, ignore_index=True )
                qj_index.append( hash )
        qj_dates = qj_dates.fillna(-1) # -1s for NaNs
        if self.verbose:
            print( qj_dates.head() )
        #
        # Convert to QJ object and return those?
        #
        return qj_dates
    

def sep(x):
    #print( "-" * 40, x )
    pass

def quux(combined, column, newname):
    '''
    TODO: Some edge cases, starting with 0, ending with 0, holiday period

    This is like language model smoothing, mabe take half of
    first non-zero value and spread it out?
    '''
    zeroes  = 0
    mangled = combined[column].tolist()
    print( mangled )
    for i, row in enumerate(combined.itertuples()):
        v = getattr(row, column)
        #print( i, v )
        mangled[i] = v
        if v == 0:
            zeroes += 1
            continue
        if zeroes > 0 and v > 0: # fill backwards with mean
            mean = v / (zeroes+1)
            #print( "mean", mean, "i", i, "zeroes", zeroes )
            #print( [ mean ] * (zeroes+1)  )
            mangled[ i-zeroes:i+1] = [ mean ] * (zeroes+1) 
            zeroes = 0
    # add new column
    se = pd.Series(mangled)
    combined[newname] = se
    #print( combined )

def smooth(combined, column, newname):
    '''
    TODO: Some edge cases, starting with 0, ending with 0, holiday period

    This is like language model smoothing, mabe take half of
    first non-zero value and spread it out?
    '''
    zeroes  = 0
    #mangled = combined[column]
    combined[newname] = combined[column]
    i = 0
    for idx, row in combined.iterrows():
        #print( i, row[column] )
        #mangled[i] = row.CLAIMCOUNT
        combined.iloc[i][newname] = row[column]
        if row[column] == 0:
            zeroes += 1
            i += 1
            continue
        if zeroes > 0 and row[column] > 0: # fill backwards with mean
            mean = row[column] / (zeroes+1)
            #print( "mean", mean, "i", i, "zeroes", zeroes )
            #mangled[ i-zeroes:i+1] = mean
            combined[newname].iloc[i-zeroes:i+1] = mean #note order for assignment
            zeroes = 0
        i += 1
    #print( combined )

def select_between_dates(df, d0, d1):
    res = df[ (df.index >= d0) & (df.index < d1) ]
    #res = res.dropna()
    return res

def date_diff(d0, d1):
    return int(str(d1 - d0).split()[0]) + 1
        
# dates here? or just data? on the whole data, so supply 90 days
def calculate_hitrate(combined, colname):
    if combined.empty:
        #print( "WARNING: trying to calculate hitrate on empty dataframe ("+colname+")." )
        return 0, 0
    D0 = combined.index[0]  # date of first
    D1 = combined.index[-1] # date of last
    days = int(str(D1 - D0).split()[0]) + 1
    #print( "days:", days )
    values = combined[colname].values.astype('float64')
    mean   = np.nan
    ave    = np.nan
    if len(values):
        #print( "values:", len(values) )
        mean = values.mean()
        ave  = values.sum() / days
    return mean, ave

# NOT USED:
class QJ():
    def __init__(self, idx):
        self.idx = idx
        self.data_before = None
        self.data_during = None
        self.data_after  = None
        self.fgrp4    = None
        self.qj_open  = None
        self.qj_close = None
        self.pe_start = None # period starts (qj open - days)
        self.pe_end   = None
        self.pe_days  = 0
        self.values_before = None
    def __str__(self):
        #      "idx      open      close     days     mean before during after     ratio      f_effect m_effect"
        return "{0:2n} | {1:10s} | {2:10s} | {3:4n}".format(
            self.idx,
            str(self.qj_open)[0:10],
            str(self.qj_close)[0:10],
            self.get_qj_duration())
    def set_data_before(self, df):
        self.data_before = df.copy()
    def set_data_during(self, df):
        self.data_during = df.copy()
    def set_data_after(self, df):
        self.data_after = df.copy()
    def set_dates(self, o, c):
        self.qj_open  = o
        self.qj_close = c
    def set_period(self, s, e, d):
        self.pe_start = s # start of period
        self.pe_end   = e # end of period
        self.pe_days  = d # duration of period in days (typically 90)
    def get_qj_duration(self):
        return int(str(self.qj_close - self.qj_open).split()[0]) + 1

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print( "Use this for import" )
    foo = DATA()
    foo.read_full()
    sep( "-"*40 )
    qjs, claims, combined, repd_combined = foo.select_fgrp( 3111 )
    print( combined )
    #
    # Taking all vehicles in a certain prod month, look/plot the claims
    #
    prod_date = "2014-08"
    prod_dt   = datetime.strptime(prod_date, '%Y-%m')
    quux = foo.asmbvol[ (foo.asmbvol['VEH_ASSEMB_MONTH'] == prod_date) ]
    print( quux[["CHASSIS_ID", "VEH_ASSEMB_MONTH"]] )
    chassis_ids = pd.unique(quux["CHASSIS_ID"])
    print( chassis_ids, len(chassis_ids) )
    # need a "get claims per chassis-id? join/merge, subselect. These are frgp4 3111 ^^
    #print( claims.head(), list(claims.columns) )
    new_df     = pd.DataFrame( columns=["CHASSIS_ID", "DT", "FGRP4"] )
    new_df_lst = []
    num_chassis_with_claim = 0
    for cis in chassis_ids:
        c_claims = claims[(claims["PRODUCT_ID"]==cis)]
        if not c_claims.empty:
            num_chassis_with_claim += 1
            #print( cis, len(c_claims) ) #c_claims[["CLAIM_REG_DATE", "FGRP4"]].values )
            for i, row in c_claims.iterrows():
                new_df_lst.append( [row["CHASSIS_ID"], row["CLAIM_REG_DATE"], row["FGRP4"] ] )
    #print( new_df_lst )
    new_df = pd.DataFrame( new_df_lst, columns=["CHASSIS_ID", "DATE", "FGRP4"] )
    #new_df["DT"] = pd.to_datetime(new_df["DATE"]) #not needed, already datetime64[ns]
    new_df = new_df.sort_values(by='DATE')
    print( new_df )
    #print( new_df.groupby(new_df["DATE"]).count() )
    #print( new_df.groupby( pd.Grouper(key='DATE', freq='D') ).count() ) #best
    #plotdata = new_df.groupby( pd.Grouper(key='DATE', freq='D') ).count()
    ##plotdata = pd.DataFrame( {'count' : new_df.groupby( ["DATE"] ).size() }).reset_index() #works, next better
    plotdata = pd.DataFrame( {'count' : new_df.groupby( pd.Grouper(key='DATE', freq='M') ).size() }).reset_index()
    #                                                                        , freq='D' #fills zeroes
    plotdata = plotdata.set_index("DATE")
    #print( plotdata )
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
    #fig.subplots_adjust(bottom=0.3)
    #
    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_locator(months) #years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.set_ylim(0, max(plotdata["count"])+1)
    ax.set_xlim( prod_dt, prod_dt + timedelta(days=760) )
    #
    days_in_month = [(plotdata.index[j+1]-plotdata.index[j]).days * -0.8 for j in range(len(plotdata.index)-1)] \
        + [30*-0.8]
    ax.bar( plotdata.index, plotdata["count"],
            width=days_in_month, #24, #in "days"
            align="edge",
            #edgecolor="black",
    )
    '''
    ax.vlines(x=plotdata.index,
              ymin=0,
              ymax=plotdata["count"],
              color="b",
              alpha=0.5,
              #linewidth=2,
              label="Num Claims")
    '''
    #plotdata.plot( ax=ax )
    fig.autofmt_xdate()
    #
    #ax.xaxis.label.set_size(8)
    plt.xticks( fontsize=8 )
    #
    ax.set_xlabel( 'Number of claims FGRP'+str(2584) )
    ax.set_title( str(len(chassis_ids))+" / "+prod_date+" / "+str(num_chassis_with_claim) )
    #
    foo.totvol.plot() # this plot volume and warranty volume
    foo.monvol["Sample Size"].plot() #green line
    print( foo.monvol["Sample Size"] )
    #
    plt.show(block=True)
    #
    #print( list(foo.asmbvol.columns) )
    #print( claims[ (claims.index < "2013-04-01") & (claims.index >= "2013-03-01") ] )
    sys.exit(1)
    qjs = foo.select_qjs_fgrp( 3111 )
    print( qjs )
    print( calculate_hitrate( combined, 'TOTNORMALISED' ))
    print( calculate_hitrate( repd_combined, 'TOTNORMALISED' ))

    sep( "-"*40 )
    
    qjs, claims, combined, repd_combined = foo.select_fgrp( 2846 )
    qjs = foo.select_qjs_fgrp( 2846 )
    print( qjs )
    for index, row in qjs.iterrows():
        #the_qj = QJ(index)
        qj_0   = row["qj_open"]
        qj_1   = row["qj_close"]
        fe     = row["f_effect"]
        me     = row["m_effect"]
        dt_t   = timedelta(days=90) 
        dt_0   = qj_0 - dt_t  # start of days period before QJ open
        dt_1   = qj_1 + dt_t  # end of days period after QJ close
        print( dt_0, qj_0, date_diff(qj_0, qj_1), qj_1, dt_1 )
        qq = select_between_dates( combined, dt_0, qj_0 )
        print( calculate_hitrate( qq, 'TOTNORMALISED' ))
        #the_qj.calculate() # Calculate mean hitrate over "total volume normalised" claim rates
        #ax.plot( (dt_0, qj_0),
        #         #(self.mean_before, self.mean_before),
        #         linewidth=4,
        #         color=c)
    print( calculate_hitrate( combined, 'TOTNORMALISED' ))
    print( calculate_hitrate( repd_combined, 'TOTNORMALISED' ))
