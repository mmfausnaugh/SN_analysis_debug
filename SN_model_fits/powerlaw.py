import numpy as np
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

def merit_with_prior(params, func, x, y, ey):
    #prior = 1./params['index'].value
    if params['index'].value < 0.5:
        prior = 1.e30
    else:
        prior = 0
    return np.sum((y - func(x,params))**2/ey**2) + prior

def merit_with_scale_prior(params, func, x, y, ey):
    #prior = 1./params['index'].value
    prior = 1./params['norm'].value
    #    print(params['norm'].value,prior)
    #print(np.sum((y - func(x,params))**2/ey**2) + prior)
    return np.sum((y - func(x,params))**2/ey**2) + prior

def powerlaw(tin,params):
    #fits for a time of explosion and a power-law rise
    #assumes a single sector, with flux calibrated to zero at early times

#    print([(key,params[key].value) for key in params.keys()])
    
    #tin should be a single array of times
    out = sp.zeros(len(tin))
    #power law takes effect oafter t_explosion
    m = tin > params['t_explosion']
    out[m] = params['norm']*(tin[m] - params['t_explosion'])**params['index']
    
    return out

def powerlaw_with_baseline(tin,params):
    #fits for a time of explosion, power-law rise, and early time flux.
    #assumes a single sector, with flux calibrated to zero at early times

#    print([(key,params[key].value) for key in params.keys()])
    
    #tin should be a single array of times
    out = sp.zeros(len(tin)) + params['baseline']
    #power law takes effect oafter t_explosion
    m = tin > params['t_explosion']
    out[m] = params['norm']*(tin[m] - params['t_explosion'])**params['index'] + params['baseline']
    
    return out


def fit_powerlaw_to_single_sector(times, fluxes,efluxes):
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
    pinit.add('norm',  value= flux0.mean(), min=0)
    pinit.add('index', value= 2.0,min=0.0, max=4.0)
    
    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
    mresult = minimize(merit, pinit, args = (powerlaw,
                                             time0, flux0, eflux0) )
    return mresult

def fit_powerlaw_with_baseline_to_single_sector(times, fluxes,efluxes):
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
    if flux0.mean() < 0:
        pinit.add('norm',  value= 0.01, min=1.e-32)
    else:
        pinit.add('norm',  value= flux0.mean(),min=1.e-32)
    pinit.add('index', value= 2.0,min=0.0, max=4.0)
    pinit.add('baseline', value= 0.0)
    
    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
    mresult = minimize(merit, pinit, args = (powerlaw_with_baseline,
                                             time0, flux0, eflux0) )
    return mresult

def fit_powerlaw_with_baseline_to_single_sector_fixed_t_explosion(times, fluxes,efluxes, t_explosion):
    #wrapper to fit a single power law and t_explosion to a single sector of data
    #imposes initial values and limits
    
    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit
    time0 = times[0]
    flux0 = fluxes[0]
    eflux0 = efluxes[0]

    #to do!! where will initial values and limits come from?
    pinit = Parameters()
    pinit.add('t_explosion',value=t_explosion,  vary=False)
    pinit.add('norm',  value= flux0.mean(),min=1.e-32)
    pinit.add('index', value= 2.0,min=0.5, max=4.0)
    pinit.add('baseline', value= 0.0)
    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
    mresult = minimize(merit, pinit,
                       args = (powerlaw_with_baseline,
                               time0, flux0, eflux0),
                       )
#    mresult = minimize(merit_with_scale_prior, pinit,
#                       args = (powerlaw_with_baseline,
#                               time0, flux0, eflux0),
#                       method='lbfgsb',
#                       )
    return mresult


def powerlaw_with_calibration(tin, params):
    #fits for a time of explosion and a power-law rise, and allows early time to have nonzero baseline
    #assumes a two sectors---sector 1 should be calibrated to zero at early times
    #sector 2 can have an offset, which is calibrated with the fit

    #tin should be a 2 element list, with the first then the second sector
    tin1 = tin[0]
    tin2 = tin[1]

    #note that out2 has it's own caliration offset
    out1 = sp.zeros(len(tin1)) + params['baseline']
    out2 = sp.zeros(len(tin2))
    
    m = tin1 > params['t_explosion']
    out1[m] = params['norm']*(tin1[m] - params['t_explosion'])**params['index'] + params['baseline']
    m = tin2 > params['t_explosion']
    out2[m] = params['norm']*(tin2[m] - params['t_explosion'])**params['index'] + params['baseline'] + params['offset']
    return sp.r_[out1,out2]


def fit_powerlaw_to_two_sectors(times, fluxes, efluxes):
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
    pinit.add('norm',  value= 1.0)
    pinit.add('index', value= 1.0,min=0.0, max=4.0)
    pinit.add('baseline', value=0.0)
    pinit.add('offset', value= flux2.min() )
    
    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
    mresult = minimize(merit, pinit,
                       args = (powerlaw_with_calibration,
                                             [time1, time2],
                                             sp.r_[flux1, flux2],
                                             sp.r_[eflux1,eflux2]),)
    return mresult


