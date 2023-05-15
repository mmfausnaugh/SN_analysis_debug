import numpy as np
import scipy as sp
from scipy.stats import kstest
import os
import sys
import glob
from copy import deepcopy

import dynesty
from dynesty import plotting as dyplot

from SN_model_fits.curved_powerlaw import prepare_data


import SN_model_fits.result_params as result_params

from SN_model_fits.kasen_2010 import *

                

if __name__ == "__main__":

    #for this light curve, the fit is OK in the new dynest, but
    #but the acceptance fraciton is low and the sampling is slow

    ##################################################
    #load data, get all the functions needed for dynesty
    ##################################################
    fit_times,fit_fluxes,fit_efluxes = np.genfromtxt('test_lc_2020tld.txt',unpack=1)
    with open('test_2020tld_params.txt','r') as fin:
        redshift = float(fin.readline())
        first_light = float(fin.readline())
    log_likelihood, prior_transform, ndim,\
        func_use, time0, flux0, eflux0, \
        redshift, labels, \
        lc_interp_array,error_norm = prepare_data([fit_times],
                                                  [fit_fluxes],
                                                  [fit_efluxes],
                                                  redshift=redshift,
                                                  first_light = first_light,
                                                  fit_companion = False,
                                                  lc_interp_array = [] )

    ##################################################
    #Run dynesty
    ################################################
    dns = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim,
                                            logl_args=[func_use,
                                                       time0, flux0, eflux0,
                                                       redshift, labels, lc_interp_array,
                                                       error_norm],

                                            bootstrap=0)

    dns.run_nested(maxcall=3.e6)
    result = dns.results

    

    ##################################################
    #process outputs, save results, make plots, flag if posteriors seem too narrow
    ################################################
    np.savez('test_2020tld_dynesty_results.npz',result)


    
    PL_result = result_params.PL_Result( ['t_explosion',
                                          'norm',
                                          'index1',
                                          'index2',
                                          'baseline'],
                                         result,
                                         separations = [],
                                         radii = [])
    PL_result.params['redshift'] = result_params.ResultParam(redshift,
                                                             0,0,0)
    

    result_params.process_results(PL_result,'test_output_2020tld.txt')

    #make the plots
    labels = ['First Light $t_0$',
              'Normalization C',
              'PL Index $\\beta_1$',
              'PL Index $\\beta_2$',
              'Background $f_0$']
    fig,axes = dyplot.cornerplot(result,
                                 #show_titles=True,
                                 #smooth=50, #defines Nbins
                                 labels=labels,
                                 show_titles=False)
    plt.gcf().suptitle('SN2020tld',fontsize=28)
    plt.savefig('test_2020tld_corner.png')

    fig,axes = dyplot.traceplot(result,
                                show_titles=True,
                                smooth=50,
                                labels=labels)
    plt.gcf().suptitle('SN2020tld')
    plt.savefig('test_2020tld_trace.png')
    

    if PL_result.params['index1'].stderr < 0.1:
        print('There may be an issue, posterior distribution'
              ' seems too narrow, with standard deviation of'
              ' index 1 = {}, should be greater than 0.1'.format(PL_result.params['index1'].stderr))

    if PL_result.params['t_explosion'].stderr < 0.1:
        print('There may be an issue, posterior distribution'
              ' seems too narrow, with standard deviation of'
              ' t explosion = {}, should be greater than 0.1'.format(
                  PL_reult.params['t_explosion'].stderr))


        
    
    #####################################################
    #Test #2
    ####################################################

    #for this light curve, the acceptance fraction is low and the
    #posterior is not sampled very well
    fit_times1,fit_fluxes1,fit_efluxes1 = np.genfromtxt('test_lc_2022exc_s1.txt',unpack=1)
    fit_times2,fit_fluxes2,fit_efluxes2 = np.genfromtxt('test_lc_2022exc_s2.txt',unpack=1)
    fit_times = [fit_times1,fit_times2]
    fit_fluxes  = [fit_fluxes1, fit_fluxes2]
    fit_efluxes = [fit_efluxes1,fit_efluxes2]
    
    with open('test_2020tld_params.txt','r') as fin:
        redshift = float(fin.readline())
        first_light = float(fin.readline())

    log_likelihood, prior_transform, ndim,\
        func_use, time0, flux0, eflux0, \
        redshift, labels, \
        lc_interp_array, error_norm= prepare_data(fit_times,
                                                  fit_fluxes,
                                                  fit_efluxes,
                                                  redshift=redshift,
                                                  first_light = first_light,
                                                  fit_companion = False,
                                                  lc_interp_array = [] )
    dns = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim,
                                            logl_args=[func_use,
                                                       time0, flux0, eflux0,
                                                       redshift, labels, lc_interp_array,
                                                       error_norm],
                                            bootstrap=0)

    dns.run_nested(maxcall=3.e6)
    result = dns.results

    np.savez('test_2022exc_dynesty_results.npz',result)
    
    PL_result = result_params.PL_Result( ['t_explosion',
                                          'norm',
                                          'index1',
                                          'index2',
                                          'baseline',
                                          'offset'],
                                         result,
                                         separations = [],
                                         radii = [])
    PL_result.params['redshift'] = result_params.ResultParam(redshift,
                                                             0,0,0)
    
    result_params.process_results(PL_result,'test_output_2022exc.txt')
    #make the plots
    labels = ['First Light $t_0$',
              'Normalization C',
              'PL Index $\\beta_1$',
              'PL Index $\\beta_2$',
              'Background $f_0$',
              'S2 offset']

    fig,axes = dyplot.cornerplot(result,
                                 #show_titles=True,
                                 #smooth=50, #defines Nbins
                                 labels=labels,
                                 show_titles=False)
    plt.gcf().suptitle('SN2020tld',fontsize=28)
    plt.savefig('test_2022exc_corner.png')
    plt.savefig('test_2022exc_corner.png')

    fig,axes = dyplot.traceplot(result,
                                show_titles=True,
                                smooth=50,
                                labels=labels)
    plt.gcf().suptitle('SN2022exc')
    plt.savefig('test_2022exc_trace.png')

    if PL_result.params['index1'].stderr < 0.1:
        print('There may be an issue, posterior distribution'
              ' seems too narrow, with standard deviation of'
              ' index 1 = {}, should be greater than 0.1'.format(PL_result.params['index1'].stderr))

    if PL_result.params['t_explosion'].stderr < 0.1:
        print('There may be an issue, posterior distribution'
              ' seems too narrow, with standard deviation of'
              ' t explosion = {}, should be greater than 0.1'.format(
                  PL_reult.params['t_explosion'].stderr))
