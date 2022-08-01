import numpy as np
import scipy as sp
from scipy.stats import kstest
import os
import sys
import glob
from copy import deepcopy

from SN_model_fits.curved_powerlaw import fit_curved_powerlaw_to_single_sector, fit_curved_powerlaw_to_two_sectors, curved_powerlaw_with_calibration, curved_powerlaw_with_baseline, fit_curved_powerlaw_to_two_sectors, dnest_curved_powerlaw

from SN_model_fits.kasen_2010 import *


FIT_COMPANION = False

class ResultParam(object):
    #quick hack to make the output of dynesty more like output of lmfit
    def __init__(self, value, stderr,lim_3sig_low,lim_3sig_high):
        self.value = value
        self.stderr = stderr
        self.lim_3sig_low  = lim_3sig_low
        self.lim_3sig_high = lim_3sig_high

class PL_Result(object):
    #take the dynesty dictionary and populates attributes used for the
    #make_plots and make_tables functions
    def __init__(self,params, input_dict, separations=None,radii=None):
        self.results = input_dict
        self.params = {}
        self.max_companion_params = {}
        weights = np.exp(self.results['logwt'] - self.results['logz'][-1])
        #print('check sum of weights:',np.sum(weights))
        for ii,p in enumerate(params):
            #assumes order in params list is the same as in results
            #array

            #found this equation in the dynesty.plotting.cornerplot
            #code; matches definition of importance weights in
            #"Posterior Estimation"
            #here:https://dynesty.readthedocs.io/en/latest/overview.html
            

            #to get the weights for percgtile calcs done correclty
            idx = np.argsort(self.results['samples'][:,ii])
            cdf = np.cumsum(weights[idx])
            cdf /= cdf[-1]
            #need a .value and a .stderr
            #value = np.mean(weights*self.results['samples'][:,ii])
            #median and 2 sigma
            ll,l,value,h,hh = np.interp([0.0013,0.025, 0.5, 0.975,0.9987],
                            cdf,
                            self.results['samples'][:,ii][idx] )

            
            stderr = (h-l)##/4.0 #+2 sigma to -2 sigma
            #print(p,value, l, h, stderr)
            self.params[p] = ResultParam(value , stderr,ll,hh)
        self.logz = self.results['logz']
        #these errors are underestimated compared to bootstrapping the
        #path of live points
        self.logzerr = self.results['logzerr']

        if FIT_COMPANION:
            if 'offset' in params:
                ii = 6
            else:
                ii = 5
            
            #to get the weights for percgtile calcs done correclty
            idx = np.argsort(self.results['samples'][:,ii])
            cdf = np.cumsum(weights[idx])
            cdf /= cdf[-1]
            #need a .value and a .stderr
            #value = np.mean(weights*self.results['samples'][:,ii])
            #median and 2 sigma
            l,value,h, highest = np.interp([0.025, 0.5, 0.975, 0.999],
                            cdf,
                            self.results['samples'][:,ii][idx] )

            l = int(l)
            value = int(value)
            h = int(h)
            highest = int(highest)
            #print('companion_index', l, value, h)
            #print(separations, l,value,h)
            #print('separations',separations[l], separations[value],
            #      separations[h], separations[highest])
            #print('radius',radii[l], radii[value],
            #      radii[h], radii[highest])

            self.med_idx = value
            self.med_separation = separations[value]*1.e13
            self.med_radius = radii[value]

            self.stderr_separation = (separations[h] - separations[l])*1.e13#(separations[h] - separations[l])/4.0*1.e13
            self.stderr_radius = (radii[h] - radii[l])/4.0

            self.max_idx = highest
            self.max_separation = separations[highest]*1.e13
            self.max_radius = radii[highest]

            
            m = self.results['samples'][:,ii][idx] == self.max_idx
            for jj,p in enumerate(params):
                cdf = np.cumsum(weights[idx][m])
                cdf /= cdf[-1]
                #just want the median
                med = np.interp([0.5],cdf,
                                self.results['samples'][:,jj][idx][m] )
                
                self.max_companion_params[p] = med
                
