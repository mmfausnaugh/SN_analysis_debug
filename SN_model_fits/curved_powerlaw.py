import numpy as np
from numpy.linalg import det
import scipy as sp
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import glob
import os
import sys
from copy import deepcopy
from scipy.interpolate import interp1d

from lmfit import minimize, Parameters

from emcee import EnsembleSampler
import dynesty
from dynesty import plotting as dyplot

import corner
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

def log_prob(pin, func, x, y, ey, redshift):
    prior = 0
    if (pin[0] < x.min()) or (pin[0] > x.max()):
        prior += -np.inf

    if pin[1] < 0:
        prior += -np.inf
        
    if pin[2] > 4.0:
        prior += -np.inf
    if pin[2] < 0.0:
        prior += -np.inf
        
    if (pin[3] < -0.10) or (pin[3] > 0.0):
        prior += -np.inf


    params = {'t_explosion':pin[0],
              'norm':pin[1],
              'index1':pin[2],
              'index2':pin[3],
              'baseline':pin[4],
              'redshift':redshift}
    return -0.5*np.sum(  (y - func(x,params))**2/ey**2) + prior

def log_likelihood(pin, func, x, y, ey, redshift, labels, lc_interp_array, error_norm):
    #used for dynesty
    params = {}
    for z in zip(labels,pin):
        params[z[0]] = z[1]
    params['redshift'] = redshift

    if 'companion_index' in params.keys():
        params['companion_index'] = int(params['companion_index'])
    
    if len(x) == 2:
        yuse  = np.r_[y[0],y[1]]
        eyuse = np.r_[ey[0],ey[1]]
    else:
        yuse = y
        eyuse = ey

    #multiply normalization by number of free params, which is what
    #is in params dict - 1 for redshift

    #plotting to check
    #plot_y = func(x,params, lc_interp_array)
    #plt.plot(x,yuse,'k.')
    #plt.plot(x,plot_y,'b-')
    #companion_lc = lc_interp_array[params['companion_index']](x - params['t_explosion'])
    #plt.plot(x,companion_lc ,'r')
    #plt.plot(x,plot_y - companion_lc ,'m')
    #plt.show()
    #chi2 + log(  1./sqrt(2*pi^N + det(covar)) )
    #for a diagonal matrix, multiply the diagonal terms together to get the determinate
    #print('chi2',-0.5*np.sum(  (yuse - func(x,params, lc_interp_array))**2/eyuse**2))
    #print('2pi factor',-0.5* ( len(eyuse)*np.log( 2*np.pi)))
    #print('summin glogs', 2*np.sum(np.log(eyuse)))
    #print('det',min(eyuse**2), max(eyuse**2), np.prod(eyuse**2) )

    #print(np.c_[x[0][0:10], x[1][0:10],
    #            yuse[0:10], eyuse[0:10],
    #            func(x,params,lc_interp_array)[0:10],
    #            ])
    #print(-0.5* ( len(eyuse)*np.log( 2*np.pi) + 2*error_norm ))
    #print(-0.5* ( len(eyuse)*np.log( 2*np.pi)),  -0.5*2*error_norm, eyuse[0] )
    #print( -0.5*np.sum(  (yuse - func(x,params, lc_interp_array))**2/eyuse**2))

    return -0.5*np.sum(  (yuse - func(x,params, lc_interp_array))**2/eyuse**2)  + \
        -0.5* ( len(eyuse)*np.log( 2*np.pi) + 2*error_norm )

def log_likelihood2(pin, func, x, y, ey, redshift):
    params = {'t_explosion':pin[0],
              'norm':pin[1],
              'index1':pin[2],
              'index2':pin[3],
              'baseline':pin[4],
              'offset':pin[5],
              'redshift':redshift}
    #multiply normalization by number of freem params, which is what
    #is in params dict - 1 for redshift

    return -0.5*np.sum(  (np.r_[y[0],y[1]] - func(x,params))**2/np.r_[ey[0],ey[1]]**2)  + \
        -0.5*np.log( 2*np.pi)*(len(params.keys()) -1)

def log_likelihood_gp(pin, func, x, y, ey, redshift, labels, lc_interp_array):
    #used for dynesty
    params = {}
    for z in zip(labels,pin):
        params[z[0]] = z[1]
    params['redshift'] = redshift

    if 'companion_index' in params.keys():
        params['companion_index'] = int(params['companion_index'])
    
    if len(x) == 2:
        yuse  = np.r_[y[0],y[1]]
        eyuse = np.r_[ey[0],ey[1]]
    else:
        yuse = y
        eyuse = ey

    #multiply normalization by number of free params, which is what
    #is in params dict - 1 for redshift

    #plotting to check
    #plot_y = func(x,params, lc_interp_array)
    #plt.plot(x,yuse,'k.')
    #plt.plot(x,plot_y,'b-')
    #companion_lc = lc_interp_array[params['companion_index']](x - params['t_explosion'])
    #plt.plot(x,companion_lc ,'r')
    #plt.plot(x,plot_y - companion_lc ,'m')
    #plt.show()
    residuals = yuse - func(x,params, lc_interp_array)
    dt = x - x.reshape(  (len(x),1)   )
    covar = np.exp(-(abs(dt))/params['tau_memory'])*eyuse**2
    covar[  abs(dt) > 2*params['tau_memory'  ] ] = 0.0
    covar = np.matrix(covar)

    #multivariate guassian is
    #exp( -0.5 res.T*covar.I*res)* 1./ sqrt( 2pi^k det(covar) )
    return -0.5*(residuals*(covar.I*residuals.reshape( (len(x),1) ) )) +\
        -0.5*(len(residuals)*np.log( 2*np.pi) + det(covar) )


def log_prob_2sectors(pin, func, x, y, ey, redshift):
    prior = 0
    if (pin[0] < x[0].min()) or (pin[0] > x[1].max()):
        prior += -np.inf

    if pin[1] < 0:
        prior += -np.inf
        
    if pin[2] > 4.0:
        prior += -np.inf
    if pin[2] < 0.0:
        prior += -np.inf
        
    if (pin[3] < -0.10) or (pin[3] > 0.0):
        prior += -np.inf


    params = {'t_explosion':pin[0],
              'norm':pin[1],
              'index1':pin[2],
              'index2':pin[3],
              'baseline':pin[4],
              'offset':pin[5],
              'redshift':redshift}
    return -0.5*np.sum(  (y - func(x,params))**2/ey**2) + prior

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


def curved_powerlaw_with_baseline(tin,params, *args):
    #fits for a time of explosion, power-law rise, and early time flux.
    #assumes a single sector, with flux calibrated to zero at early times

#    print([(key,params[key].value) for key in params.keys()])
    
    #tin should be a single array of times
    
    out = np.zeros(len(tin)) + params['baseline']
    
    #power law takes effect oafter t_explosion
    delta_t = (tin - params['t_explosion'])/( 1. + params['redshift'])
    m = delta_t > 0    
    exponent = params['index1']*( 1+ params['index2']*(delta_t[m]))
    
    out[m] = params['norm']*(delta_t[m])**exponent + params['baseline']
    
    return out

###########################################
####**********DEVELOP!!!!!!!!!!!!!!!!!!!
#######################################
def curved_powerlaw_with_companion(tin,params,lc_interp_array):
    #fits for a time of explosion, power-law rise, and early time flux.
    #assumes a single sector, with flux calibrated to zero at early times

#    print([(key,params[key].value) for key in params.keys()])
    
    #tin should be a single array of times

    lc_interp = lc_interp_array[params['companion_index']]
    out = np.zeros(len(tin)) + params['baseline']
    
    #power law takes effect oafter t_explosion
    delta_t = (tin - params['t_explosion'])/( 1. + params['redshift'])
    m = delta_t > 0    
    exponent = params['index1']*( 1+ params['index2']*(delta_t[m]))
    
    out[m] = params['norm']*(delta_t[m])**exponent + params['baseline'] + lc_interp(delta_t[m])

    
    return out



def fit_curved_powerlaw_to_single_sector(times, fluxes,efluxes, redshift = 0, fixed_t_exp = None):
    #wrapper to fit a single power law and t_explosion to a single sector of data
    #imposes initial values and limits
    
    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit
    time0 = times[0]
    flux0 = fluxes[0]
    eflux0 = efluxes[0]

    #to do!! where will initial values and limits come from?
    pinit = Parameters()
    if fixed_t_exp is not None:
        pinit.add('t_explosion',value=fixed_t_exp, vary = False)
        
    else:
        pinit.add('t_explosion', value = time0.min()+1,  min = time0.min(), max = time0.max())
        
    if flux0.mean() < 0:
        pinit.add('norm',  value= 0.01, min=1.e-32)
    else:
        pinit.add('norm',  value= flux0.mean(),min=1.e-32)

    pinit.add('index1', value= 2.0,min=0.0, max=100.0)
    pinit.add('index2', value=-0.00001,min=-0.10, max=0.0)
    pinit.add('baseline', value= 0.0)
    pinit.add('redshift', value = redshift, vary=False)
    
    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
    mresult = minimize(merit, pinit, args = (curved_powerlaw_with_baseline,
                                             time0, flux0, eflux0),
                       ftol=1.e-12,
                       xtol=1.e-12)
    return mresult



def mcmc_curved_powerlaw_to_single_sector(times, fluxes,efluxes, redshift = 0,
                                          fixed_t_exp = None,
                                          nwalk = 30,
                                          niter=175000,
                                          discard=25000):


    np.random.seed(101010)
    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit
    time0 = times[0]
    flux0 = fluxes[0]
    eflux0 = efluxes[0]


    #order in param vector:
    #t_explosion, norm, index1, index2, baseline
    #redshift is fixed
    p0  = []
    if fixed_t_exp is not None:
        p0.append(fixed_t_exp)
    else:
        p0.append(time0.mean())
        
    if flux0.mean() < 0:
        p0.append( 0.01)
    else:
        p0.append( flux0.mean() )
    p0.append(2.0)
    p0.append(-0.001)
    p0.append(0.0)

    p0 = np.array(p0)
    var_inits = np.random.randn(nwalk,5)#*1.e-4
    #var_inits[:,0] *= 1.  #good scale to start days
    var_inits[:,1] *= 1.e-2
    var_inits[:,2] *= 5.e-1
    var_inits[:,3] *= 1.e-5
    var_inits[:,4] *=1.e-1
    p0 = p0 + var_inits
    p0[:,1][p0[:,1] < 0] = 0.01
    p0[:,2][p0[:,2] > 4.0] = 3.9
    p0[:,2][p0[:,2] < 0.0] = 0.001
    p0[:,3][p0[:,3] < -0.1] = -0.09
    p0[:,3][p0[:,3] > 0.0 ] = -0.000001
   
    nwalkers, ndim = p0.shape

    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
        #    def log_prob(params, func, x, y, ey):
    sampler = EnsembleSampler(nwalkers,ndim,log_prob, args=[curved_powerlaw_with_baseline,
                                                            time0,
                                                            flux0,
                                                            eflux0,
                                                            redshift] )
    sampler.run_mcmc(p0, niter, progress=True)
    dist = sampler.get_chain(discard=discard)
    print(np.shape(dist))

    #plot here, get perctile, return
    d1 = np.ravel(dist[:,:,0])
    d2 = np.ravel(dist[:,:,1])
    d3 = np.ravel(dist[:,:,2])
    d4 = np.ravel(dist[:,:,3])
    d5 = np.ravel(dist[:,:,4])
    dists = [d1,d2,d3,d4,d5]
    luse = ['t_exp','norm','PL1','PL2','baseline']
#    F,(axes) = plt.subplots(5,1,sharex='col')
#    for ii in range(5):
#        axes[ii].plot(dists[ii],'k-')
#        axes[ii].set_ylabel(luse[ii])
    print('acceptance fraction',sampler.acceptance_fraction)
    try:
        print('autocor time')
        print(sampler.get_autocorr_time() )
    except Exception as e:
        print(e)
        pass

    med_t_exp = np.median(d1)
    med_norm = np.median(d2)
    med_index1 = np.median(d3)
    med_index2 = np.median(d4)
    med_baseline = np.median(d5)

    out =  [med_t_exp, med_norm, med_index1, med_index2, med_baseline]
    print('testing the output',out)
    corner.corner(np.transpose(dists),labels=luse)
    plt.show()
    
    return out



def dnest_curved_powerlaw(times, fluxes,efluxes, redshift = 0,
                          fixed_t_exp = None,
                          first_light = None,
                          fit_companion=False,
                          lc_interp_array=[]):


    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit
    if len(times) == 2:
        time1  = times[0]
        len_s1 = time1[-1] - time1[0]
        print('len_s1',len_s1)
        flux1  = fluxes[0]
        eflux1 = efluxes[0]
        time2  = times[1]
        flux2  = fluxes[1]
        eflux2 = efluxes[1]

        time0 = [time1,time2]
        flux0 = [flux1,flux2]
        eflux0 = [eflux1,eflux2]
        print(time0[0][0], time0[0][-1])
        if fit_companion:
            labels = ['t_explosion',
                      'norm',
                      'index1',
                      'index2',
                      'baseline',
                      'offset',
                      'companion_index']

            n_companion =  len(lc_interp_array)
            
            prior_transform = lambda u: np.array([time0[0][0] +  u[0]*len_s1,
                                                  u[1],
                                                  4.0*u[2],
                                                  0.1*u[3] - 0.1,
                                                  0.4*u[4] - 0.2,
                                                  2.0*u[5] - 1.0,
                                                  int(n_companion*u[6])]  )
            ndim = 7
            func_use = curved_powerlaw_with_companion_with_calibration
        else:
            labels = ['t_explosion',
                      'norm',
                      'index1',
                      'index2',
                      'baseline',
                      'offset']

            prior_transform = lambda u: np.array([time0[0][0] + u[0]*len_s1,
                                                  u[1],
                                                  4.0*u[2],
                                                  0.1*u[3] - 0.1,
                                                  0.4*u[4] - 0.2,
                                                  2.0*u[5] - 1.0])
            ndim = 6
            func_use = curved_powerlaw_with_calibration

    else:
        time0 = times[0]
        len_s1= time0[-1] - time0[0]
        print('len_s1',len_s1)
        flux0 = fluxes[0]
        eflux0 = efluxes[0]
        if fit_companion:
                    
            #prior transform
            #try to make this as small as possible, will help the time it takes Dnest to work
        
            #t_explosion, norm, index1, index2, baseline
            #t_explosion between +/- 2 days of initial estimate???  estimate must be based on observed flux
            #need to prove that can be estimated reasonably
            
            #norm:  must be less than 1, but log spaced...
            #PL indeces are reasonable
            
            #baseline should be some small fraction of norm, say -0.2 to 0.2


            #companion index must be integers from 0 to len(companion_lc) - 1
            labels = ['t_explosion',
                      'norm',
                      'index1',
                      'index2',
                      'baseline',
                      'companion_index']
            n_companion =  len(lc_interp_array)
            prior_transform = lambda u: np.array([time0[0] + u[0]*len_s1,
                                                  u[1],
                                                  4.0*u[2],
                                                  0.1*u[3] - 0.1,
                                                  0.4*u[4] - 0.2,
                                                  int(n_companion*u[5])])
            ndim = 6
            func_use = curved_powerlaw_with_companion
        else:            
            labels = ['t_explosion',
                      'norm',
                      'index1',
                      'index2',
                      'baseline']
            prior_transform = lambda u: np.array([time0[0] + u[0]*len_s1,
                                                  u[1],
                                                  4.0*u[2],
                                                  0.1*u[3] - 0.1,
                                                  0.4*u[4] - 0.2 ])
            ndim = 5
            func_use = curved_powerlaw_with_baseline



    #    ln_lkly_use = lambda pin: log_likelihood(pin, curved_powerlaw_with_baseline,
    #                                             time0, flux0, eflux0, redshift)
    if len(eflux0) == 2:
        error_norm = np.sum(np.log( np.r_[eflux0[0], eflux0[1]]))
    else:
        error_norm = np.sum(np.log( eflux0[0] ))
    dsampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim,
                                            logl_args=[func_use,
                                                       time0, flux0, eflux0,
                                                       redshift, labels, lc_interp_array,
                                                       error_norm])

    #dsampler.run_nested(maxiter=10)

    #2020bj goes to .003% efficiency, and takes 2.5 days if no
    #constraint picked this value based on 2018hsz, which had the
    #largest .npz chain file on disk
    dsampler.run_nested(maxcall=3.e6)
    res = dsampler.results
#    print(res.keys())
#    res.summary()    
#    fig, axes = dyplot.runplot(res)
#    dyplot.traceplot(res,show_titles=True,smooth=200)
    
#    dyplot.cornerplot(res, show_titles=True,smooth=200)
#    plt.show()
    
    return res




def dnest_curved_powerlaw_with_companion_to_single_sector(
        times, fluxes,efluxes, redshift = 0,
        fixed_t_exp = None,
        first_light = None):


    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit
    time0 = times[0]
    flux0 = fluxes[0]
    eflux0 = efluxes[0]

    #prior transform
    #try to make this as small as possible, will help the time it takes Dnest to work

    #t_explosion, norm, index1, index2, baseline, index of companion
    #t_explosion between +/- 2 days of initial estimate???  estimate must be based on observed flux
    #need to prove that can be estimated reasonably

    #norm:  must be less than 1, but log spaced...
    #PL indeces are reasonable
    
    #baseline should be some small fraction of norm, say -0.2 to 0.2
    
    #companion index is just one for each separation/radius
    labels = ['t_explosion',
              'norm',
              'index1',
              'index2',
              'baseline',
              'companion_index']
    
    prior_transform = lambda u: np.array([first_light - 5.0 + u[0]*5.0,
                                          u[1],
                                          4.0*u[2],
                                          0.1*u[3] - 0.1,
                                          0.4*u[4] - 0.2 ,
                                          int(n_companion*u[5]) ])


#    ln_lkly_use = lambda pin: comp_log_likelihood(pin, curved_powerlaw_with_companion,
#                                             time0, flux0, eflux0, redshift)

    dsampler = dynesty.DynamicNestedSampler(ln_lkly_use, prior_transform, 6,
                                            logl_args=[curved_powerlaw_with_companion,
                                                       time0, flux0, eflux0, redshift, labels])
    dsampler.run_nested(maxiter=10)
#    dsampler.run_nested()
    res = dsampler.results
#    print(res.keys())
#    res.summary()    
#    fig, axes = dyplot.runplot(res)
#    dyplot.traceplot(res,show_titles=True,smooth=200)
    
#    dyplot.cornerplot(res, show_titles=True,smooth=200)
#    plt.show()
    
    return res

def dnest_curved_powerlaw_to_two_sectors(times, fluxes,efluxes, redshift = 0,
                                           fixed_t_exp = None,
                                           first_light = None):


    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit
    time1  = times[0]
    flux1  = fluxes[0]
    eflux1 = efluxes[0]
    time2  = times[1]
    flux2  = fluxes[1]
    eflux2 = efluxes[1]

    #prior transform
    #try to make this as small as possible, will help the time it takes Dnest to work

    #t_explosion, norm, index1, index2, baseline, offset
    #t_explosion between +/- 2 days of initial estimate???  estimate must be based on observed flux
    #need to prove that can be estimated reasonably

    #norm:  must be less than 1, but log spaced...
    #PL indices are reasonable
    
    #baseline should be some small fraction of norm, say -0.2 to 0.2

    #offset should be between -1.0 and 1.0, since dynesty will only every
    #see light curve normalized to peak.  Assumes the shift is close to within 100%
    labels = ['t_explosion',
              'norm',
              'index1',
              'index2',
              'baseline',
              'offset',]
    prior_transform = lambda u: np.array([first_light - 5.0 + u[0]*5.0,
                                          u[1],
                                          4.0*u[2],
                                          0.1*u[3] - 0.1,
                                          0.4*u[4] - 0.2,
                                          2.0*u[5] - 1.0])

    #    ln_lkly_use = lambda pin: log_likelihood2(pin, curved_powerlaw_with_calibration,
    #                                              [time1,time2],
    #                                              [flux1, flux2],
    #                                              [eflux1,eflux2],
    #                                              redshift)
    
    dsampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, 6,
                                            logl_args=[curved_powerlaw_with_calibration,
                                                       [time1,time2],
                                                       [flux1, flux2],
                                                       [eflux1,eflux2],
                                                       redshift,
                                                       labels])
#     dsampler = dynesty.DynamicNestedSampler(ln_lkly_use, prior_transform, 6)
#    dsampler.run_nested(maxiter=10)
    dsampler.run_nested()
    res = dsampler.results
#    print(res.keys())
#    res.summary()    
#    fig, axes = dyplot.runplot(res)
#    dyplot.traceplot(res,show_titles=True,smooth=200)
    
#    dyplot.cornerplot(res, show_titles=True,smooth=200)
#    plt.show()
    
    return res



def mcmc_curved_powerlaw_to_two_sectors(times, fluxes,efluxes, redshift = 0,
                                        fixed_t_exp = None,
                                        init_offset=None,
                                        nwalk=100,
                                        niter=2000):
    
    #times,fluxes, efluxes should be lists with a single element, the
    #array of values to fit
    time1  = times[0]
    flux1  = fluxes[0]
    eflux1 = efluxes[0]
    time2  = times[1]
    flux2  = fluxes[1]
    eflux2 = efluxes[1]


    #order in param vector:
    #t_explosion, norm, index1, index2, baseline, offset
    #redshift is fixed
    p0  = []
    if fixed_t_exp is not None:
        p0.append(fixed_t_exp)
    else:
        p0.append(time1.mean())
        
    if flux1.mean() < 0:
        p0.append( 0.01)
    else:
        p0.append( flux1.mean() )
    p0.append(2.0)
    p0.append(-0.001)
    p0.append(0.0)
    if init_offset is None:
        p0.append(flux2.min() *5)
    else:
        p0.append(init_offset*25)


    
    p0 = np.array(p0)
    var_inits = np.random.randn(nwalk,6)#*1.e-4
    #var_inits[:,0] *= 1.  #good scale to start days
    var_inits[:,1] *= 1.e-2
    var_inits[:,2] *= 5.e-1
    var_inits[:,3] *= 1.e-5
    var_inits[:,4] *=1.e-1
    var_inits[:,5] *= 10
    p0 = p0 + var_inits
    p0[:,1][p0[:,1] < 0] = 0.01
    p0[:,2][p0[:,2] > 4.0] = 3.9
    p0[:,2][p0[:,2] < 0.0] = 0.001
    p0[:,3][p0[:,3] < -0.1] = -0.09
    p0[:,3][p0[:,3] > 0.0 ] = -0.000001
   
    nwalkers, ndim = p0.shape

    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
        #    def log_prob(params, func, x, y, ey):
    sampler = EnsembleSampler(nwalkers,ndim,log_prob_2sectors, args=[curved_powerlaw_with_calibration,
                                                                     [time1, time2],
                                                                     np.r_[flux1, flux2],
                                                                     np.r_[eflux1,eflux2],
                                                                     redshift] )
    sampler.run_mcmc(p0, niter, progress=True)
    dist = sampler.get_chain(discard=1000)
    print(np.shape(dist))

    #plot here, get perctile, return
    d1 = np.ravel(dist[:,:,0])
    d2 = np.ravel(dist[:,:,1])
    d3 = np.ravel(dist[:,:,2])
    d4 = np.ravel(dist[:,:,3])
    d5 = np.ravel(dist[:,:,4])
    d6 = np.ravel(dist[:,:,5])
    dists = [d1,d2,d3,d4,d5, d6]
    luse = ['t_exp','norm','PL1','PL2','baseline','offset']
#    F,(axes) = plt.subplots(5,1,sharex='col')
#    for ii in range(5):
#        axes[ii].plot(dists[ii],'k-')
#        axes[ii].set_ylabel(luse[ii])
    print('acceptance fraction',sampler.acceptance_fraction)
    print('autocor time')
    print(sampler.get_autocorr_time() )
    med_t_exp = np.median(d1)
    med_norm = np.median(d2)
    med_index1 = np.median(d3)
    med_index2 = np.median(d4)
    med_baseline = np.median(d5)
    med_offset = np.median(d6)

    out =  [med_t_exp, med_norm, med_index1, med_index2, med_baseline, med_offset]
    print('testing the output',out)
    corner.corner(dists,labels=luse)
    plt.show()
    
    return out




def curved_powerlaw_with_calibration(tin, params, *args):
    #fits for a time of explosion and a power-law rise, and allows early time to have nonzero baseline
    #assumes a two sectors---sector 1 should be calibrated to zero at early times
    #sector 2 can have an offset, which is calibrated with the fit

    #tin should be a 2 element list, with the first then the second sector
    tin1 = tin[0]
    tin2 = tin[1]

    dt1 = (tin1 - params['t_explosion'])/(1. + params['redshift'])
    dt2 = (tin2 - params['t_explosion'])/(1. + params['redshift'])
    
    #note that out2 has it's own caliration offset
    out1 = np.zeros(len(tin1)) + params['baseline']
    out2 = np.zeros(len(tin2))

    
    m = dt1 > 0
    exponent = params['index1']*( 1+ params['index2']*dt1[m] )
    out1[m] = params['norm']*(dt1[m])**exponent + params['baseline']

    
    m = dt2 > 0
    exponent = params['index1']*( 1+ params['index2']*dt2[m])
    out2[m] = params['norm']*(dt2[m])**exponent + params['baseline'] + params['offset']
    return np.r_[out1,out2]

def curved_powerlaw_with_companion_with_calibration(tin, params, lc_interp_array):
    #fits for a time of explosion and a power-law rise, and allows early time to have nonzero baseline
    #assumes a two sectors---sector 1 should be calibrated to zero at early times
    #sector 2 can have an offset, which is calibrated with the fit

    
    lc_interp = lc_interp_array[params['companion_index']]
    
    #tin should be a 2 element list, with the first then the second sector
    tin1 = tin[0]
    tin2 = tin[1]

    dt1 = (tin1 - params['t_explosion'])/(1. + params['redshift'])
    dt2 = (tin2 - params['t_explosion'])/(1. + params['redshift'])
    
    #note that out2 has it's own caliration offset
    out1 = np.zeros(len(tin1)) + params['baseline']
    out2 = np.zeros(len(tin2))

    
    m = dt1 > 0
    exponent = params['index1']*( 1+ params['index2']*dt1[m] )
    out1[m] = params['norm']*(dt1[m])**exponent + params['baseline']\
              +lc_interp(dt1[m])

    
    m = dt2 > 0
    exponent = params['index1']*( 1+ params['index2']*dt2[m])
    out2[m] = params['norm']*(dt2[m])**exponent + params['baseline']\
              +lc_interp(dt2[m]) + params['offset']
    return np.r_[out1,out2]



def fit_curved_powerlaw_to_two_sectors(times, fluxes, efluxes,
                                       redshift=0,
                                       fixed_t_exp = None,
                                       init_offset=None):
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
    if fixed_t_exp is not None:
        pinit.add('t_explosion',value=fixed_t_exp, vary = False)
    else:
        #2018jwi wanted  a later guess for start time, - 6 days,  but 2018hkx wanted an earlier guess, -13
        pinit.add('t_explosion',value=time1.max() - np.random.random()*14,
                  min = time1.min() - 14., max = time1.max())
    if flux1.mean() < 0:
        pinit.add('norm',  value= 0.01, min=1.e-32)
    else:
        pinit.add('norm',  value= flux1.mean(),min=1.e-32)
    pinit.add('index1', value= 2.0, min=0.0, max=4.0)
    pinit.add('index2', value=-0.00001,min=-0.10, max=0.00)
    pinit.add('baseline', value=0.0)
    #    pinit.add('offset', value= flux2.min() )
    if init_offset is None:
        pinit.add('offset', value= flux2.min() *5)
    else:
        print('check the initial offet!!', init_offset)
        pinit.add('offset', value= init_offset )

    pinit.add('redshift', value= redshift, vary = False)

    #which part of the data to fit?
    #should be able to fit all of it...unless weird noise at early times grabs t_break
#    m1fit = time1 > time1[-100]
#    m2fit = time2 < time2[100]
    
    mresult = minimize(merit, pinit,
                       args = (curved_powerlaw_with_calibration,
                               [time1, time2],
                               np.r_[flux1, flux2],
                               np.r_[eflux1,eflux2]),
                       ftol=1.e-12,
                       xtol = 1.e-12
    )
    return mresult


