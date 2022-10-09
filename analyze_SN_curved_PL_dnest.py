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
                

if __name__ == "__main__":

    #for this light curve, the fit is OK in the new dynest, but
    #but the acceptance fraciton is low and the sampling is slow
    fit_times,fit_fluxes,fit_efluxes = np.genfromtxt('test_lc_2020tld.txt',unpack=1)
    with open('test_2020tld_params.txt','r') as fin:
        redshift = float(fin.readline())
        first_light = float(fin.readline())

    result = dnest_curved_powerlaw([fit_times],
                                   [fit_fluxes],
                                   [fit_efluxes],
                                   redshift=redshift,
                                   first_light = first_light,
                                   fit_companion = False,
                                   lc_interp_array = [] )

    np.savez('test_2020tld_dynesty_results.npz',result)

    PL_result = PL_Result( ['t_explosion',
                            'norm',
                            'index1',
                            'index2',
                            'baseline'],
                           result,
                           separations = [],
                           radii = [])
    PL_result.params['redshift'] = ResultParam(redshift,
                                               0,0,0)
    
    s2_offset_fit1 = 0

    with open('test_output_2020tld.txt','w') as fout:
        logz      = PL_result.logz[-1]
        logzerr   = PL_result.logzerr[-1]
        PL_norm   = PL_result.params['norm'].value
        PL_t_exp  = PL_result.params['t_explosion'].value
        PL_index1  = PL_result.params['index1'].value
        PL_index2  = PL_result.params['index2'].value
        PL_baseline = PL_result.params['baseline'].value

        PL_enorm   = PL_result.params['norm'].stderr
        PL_et_exp  = PL_result.params['t_explosion'].stderr
        PL_eindex1  = PL_result.params['index1'].stderr
        PL_eindex2  = PL_result.params['index2'].stderr
        PL_ebaseline = PL_result.params['baseline'].stderr

        if 'offset' in PL_result.params.keys():
            PL_offset = PL_result.params['offset'].value
            PL_eoffset = PL_result.params['offset'].stderr
        else:
            PL_offset = 0.0
            PL_eoffset = 0.0
        fout.write('{:25s}{:15.2f}\n'.format('logz',logz))
        fout.write('{:25s}{:15.2f}\n'.format('logzerr',logzerr))
        fout.write('{:25s}{:15.2f}\n'.format('normalization',PL_norm))
        fout.write('{:25s}{:15.2f}\n'.format('explosion_time',PL_t_exp))
        fout.write('{:25s}{:15.2f}\n'.format('power_law_index1',PL_index1))
        fout.write('{:25s}{:15.2f}\n'.format('power_law_index2',PL_index2))
        fout.write('{:25s}{:15.2f}\n'.format('baseline',PL_baseline))
        fout.write('{:25s}{:15.2f}\n'.format('offset',PL_offset))
        fout.write('{:25s}{:15.2f}\n'.format('error_normalization',PL_enorm))
        fout.write('{:25s}{:15.2f}\n'.format('error_explosion_time',PL_et_exp))
        fout.write('{:25s}{:15.2f}\n'.format('error_power_law_index1',PL_eindex1))
        fout.write('{:25s}{:15.2f}\n'.format('error_power_law_index2',PL_eindex2))
        fout.write('{:25s}{:15.2f}\n'.format('error_baseline',PL_ebaseline))
        fout.write('{:25s}{:15.2f}\n'.format('error_offset',PL_eoffset))


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

    result = dnest_curved_powerlaw(fit_times,
                                   fit_fluxes,
                                   fit_efluxes,
                                   redshift=redshift,
                                   first_light = first_light,
                                   fit_companion = False,
                                   lc_interp_array = [] )

    np.savez('test_2022exc_dynesty_results.npz',result)
    
    PL_result = PL_Result( ['t_explosion',
                            'norm',
                            'index1',
                            'index2',
                            'baseline',
                            'offset'],
                           result,
                           separations = [],
                           radii = [])
    PL_result.params['redshift'] = ResultParam(redshift,
                                               0,0,0)
    
    s2_offset_fit1 = 0

    with open('test_output_2022exc.txt','w') as fout:
        logz      = PL_result.logz[-1]
        logzerr   = PL_result.logzerr[-1]
        PL_norm   = PL_result.params['norm'].value
        PL_t_exp  = PL_result.params['t_explosion'].value
        PL_index1  = PL_result.params['index1'].value
        PL_index2  = PL_result.params['index2'].value
        PL_baseline = PL_result.params['baseline'].value

        PL_enorm   = PL_result.params['norm'].stderr
        PL_et_exp  = PL_result.params['t_explosion'].stderr
        PL_eindex1  = PL_result.params['index1'].stderr
        PL_eindex2  = PL_result.params['index2'].stderr
        PL_ebaseline = PL_result.params['baseline'].stderr

        if 'offset' in PL_result.params.keys():
            PL_offset = PL_result.params['offset'].value
            PL_eoffset = PL_result.params['offset'].stderr
        else:
            PL_offset = 0.0
            PL_eoffset = 0.0
        fout.write('{:25s}{:15.2f}\n'.format('logz',logz))
        fout.write('{:25s}{:15.2f}\n'.format('logzerr',logzerr))
        fout.write('{:25s}{:15.2f}\n'.format('normalization',PL_norm))
        fout.write('{:25s}{:15.2f}\n'.format('explosion_time',PL_t_exp))
        fout.write('{:25s}{:15.2f}\n'.format('power_law_index1',PL_index1))
        fout.write('{:25s}{:15.2f}\n'.format('power_law_index2',PL_index2))
        fout.write('{:25s}{:15.2f}\n'.format('baseline',PL_baseline))
        fout.write('{:25s}{:15.2f}\n'.format('offset',PL_offset))
        fout.write('{:25s}{:15.2f}\n'.format('error_normalization',PL_enorm))
        fout.write('{:25s}{:15.2f}\n'.format('error_explosion_time',PL_et_exp))
        fout.write('{:25s}{:15.2f}\n'.format('error_power_law_index1',PL_eindex1))
        fout.write('{:25s}{:15.2f}\n'.format('error_power_law_index2',PL_eindex2))
        fout.write('{:25s}{:15.2f}\n'.format('error_baseline',PL_ebaseline))
        fout.write('{:25s}{:15.2f}\n'.format('error_offset',PL_eoffset))
