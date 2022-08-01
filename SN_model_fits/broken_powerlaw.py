import scipy as sp
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import glob
import os
import sys
from copy import deepcopy
from scipy.interpolate import interp1d

from lmfit import minimize, Parameters

#started as 'calib_multisector.py'

#For each SN, designated in a dictionary with sectors observed, fits a
#powerlaw rise and a broken powerlaw, save a figure with fits, and
#save a table of parameters


#in ben's 2018oh/ASASSN-18bt, the absolute flux is set with synethic
#photometry at peak. That is worth a check

#The fits are done in linear units, so early times should be zero.
#will force that here, and convert to mags with ref flux (15,000 e/s at
#10th mag)

#return vector or residuals.  lmfit will automatically get the summed
#squares

def merit(params, func, x, y, ey):
    return (y - func(x,params))/ey

def broken_powerlaw(tin,params):
    #fits for a time of explosion and a power-law rise
    #assumes a single sector, with flux calibrated to zero at early times
    
    #tin should be a single array of times
    out = sp.zeros(len(tin))
    tbreak = params['t_explosion'] + params['delta_t']
    #power law takes effect oafter t_explosion
    m = (tin > params['t_explosion']) & (tin < tbreak)
    out[m] = params['norm']*(  (tin[m] - params['t_explosion']) /params['delta_t'] )**params['index1']
    #above tbreak, change the exponent
    m = (tin > tbreak)
    out[m] = params['norm']*(  (tin[m] - params['t_explosion']) /params['delta_t'] )**params['index2']
    
    return out


def fit_broken_powerlaw_to_single_sector(times, fluxes,efluxes):
    #wrapper to fit a single power law and t_explosion to a single sector of data
    #imposes initial values and limits
    
    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit
    time0 = times[0]
    flux0 = fluxes[0]
    eflux0 = efluxes[0]

    #to do!! where will initial values and limits come from?
    pinit = Parameters()
    pinit.add('t_explosion',value=time0.mean(),  min = time0.min(), max = time0.max())
    pinit.add('delta_t', value = 5.0,   min = 0.0, max = 100)
    pinit.add('norm',  value= 1.0)
    pinit.add('index1', value= 1.0,min=0.0, max=4.0)
    pinit.add('index2', value= 1.0,min=0.0, max=4.0)
    
    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
    mresult = minimize(merit, pinit, args = (broken_powerlaw,
                                             time0, flux0, eflux0) )
    return mresult


def broken_powerlaw_with_calibration(tin, params):
    #fits for a time of explosion and a power-law rise
    #assumes a two sectors---sector 1 should be calibrated to zero at early times
    #sector 2 can have an offset, which is calibrated with the fit

    #tin should be a 2 element list, with the first then the second sector
    tin1 = tin[0]
    tin2 = tin[1]
    out1 = sp.zeros(len(tin1))
    out2 = sp.zeros(len(tin2))

    tbreak = params['t_explosion'] + params['delta_t']

    #fit first sector
    m = (tin1 > params['t_explosion']) & (tin1 < tbreak)
    out1[m] = params['norm']*(  (tin1[m] - params['t_explosion']) /params['delta_t'] )**params['index1']
    m = (tin1 > tbreak)
    out1[m] = params['norm']*(  (tin1[m] - params['t_explosion']) /params['delta_t'] )**params['index2']

    #fit second sector
    m = (tin2 > params['t_explosion']) & (tin2 < tbreak)
    out2[m] = params['norm']*(  (tin2[m] - params['t_explosion']) /params['delta_t'] )**params['index1'] + params['offset']
    m = (tin2 > tbreak)
    out2[m] = params['norm']*(  (tin2[m] - params['t_explosion']) /params['delta_t'] )**params['index2'] + params['offset']

    
    return sp.r_[out1,out2]


def fit_broken_powerlaw_to_two_sectors(times, fluxes, efluxes):
    #wrapper to fit a single power law and t_explosion to 2 sectors, with calibration of sector 2
    #imposes initial values and limits
    
    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit

    #should assume that only the part of S2 that you want to fit is passed in here
    time1  = times[0]
    flux1  = fluxes[0]
    eflux1 = efluxes[0]
    time2  = times[1]
    flux2  = fluxes[1]
    eflux2 = efluxes[1]

    
    #to do!! where will initial values and limits come from?
    pinit = Parameters()
    pinit.add('t_explosion',value=time1.mean(),  min = time1.min(), max = time2.max())
    pinit.add('delta_t',value = 5.0, min = 0.0, max = 100)
    pinit.add('norm',  value= 1.0)
    pinit.add('index1', value= 1.0,min=0.0, max=4.0)
    pinit.add('index2', value= 1.0,min=0.0, max=4.0)
    pinit.add('offset', value =  0.0)
    
    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
    mresult = minimize(merit, pinit, args = (broken_powerlaw_with_calibration,
                                             [time1, time2],
                                             sp.r_[flux1, flux2],
                                             sp.r_[eflux1,eflux2]) )
    return mresult


