import numpy as np
import os
import sys
from copy import deepcopy

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


def process_results(PL_result,output_file):
    with open(output_file,'w') as fout:
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

        
        #if PL_eindex1 < 0.1:
        #    print('There may be an issue, posterior distribution'
        #          ' seems too narrow, with standard deviation of'
        #          ' index 1 = {}, should be greater than 0.1'.format(PL_eindex1))
        #if PL_et_exp < 0.1:
        #    print('There may be an issue, posterior distribution'
        #          ' seems too narrow, with standard deviation of'
        #          ' t explosion = {}, should be greater than 0.1'.format(PL_et_exp))
