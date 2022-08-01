import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot
from SN_model_fits.curved_powerlaw import *
import os
from analyze_SN_curved_PL_dnest import PL_Result, ResultParam

np.random.seed(1010102)
p = {'norm': 1,
     'baseline':0,
     't_explosion':0.0,
     'redshift':0,
     'index1':2.0,
     'index2':-0.015}

x = np.r_[-2:20:0.5/24]
y = curved_powerlaw_with_baseline(x,p)
y = y/y.max()
#noise_levels = np.array([0.01, 0.05,0.10, 0.2, 0.4, 0.6])
#noise_levels = np.r_[0.01:0.8:10j]
noise_levels = 10**np.r_[-2:0:10j]
print(noise_levels)

#F,(axes1,axes2) = plt.subplots(len(noise_levels)+1,1)
F = plt.figure()
gs1 = gridspec.GridSpec( len(noise_levels)+1,2)
axes = []
for ii in range(len(noise_levels)+1):
    axes.append(F.add_subplot(gs1[ii,0]) )
#this is for recovered parameter
ax2 = F.add_subplot(gs1[:,1])

PL_results = []
axes[0].plot(x,y,'k')
axes[0].text(12,0,'input index = {:.2f}'.format(2.0) )
l,u = axes[0].get_ylim()
axes[0].plot([0,0],[l,u],'k--')

for ii,noise_level in enumerate(noise_levels):
    y_sim = y + np.random.normal(0,1,len(y))*noise_level
    error = np.ones(len(x))*noise_level
    axes[ii+1].errorbar(x,y_sim,
                        error,
                        fmt='ko',
                        lw=2,
                        ms=0,
                        capsize=0)

    fit_result = fit_curved_powerlaw_to_single_sector(
        [x],[y_sim],[error],
        redshift=0,
    #    fixed_t_exp=0
    )

    chain_file = 'noise{:.2f}_sim__dynesty.npz'.format(noise_level)
    if os.path.isfile(chain_file):
        #load chain
        print('dada', chain_file)
        #f = np.load(chain_file,allow_pickle=True)
        #result = f['arr_0'].tolist()

        with np.load(chain_file,allow_pickle=True) as f:
            print(f)
            print(f.files)
            for k in f.keys():
                print(k)
            print(type(f['arr_0']))
            result = f['arr_0'].tolist()


        #print(result2)
        #result = result2['arr_0']#.tolist()
        
    else:
        result = dnest_curved_powerlaw(
            [x],[y_sim],[error],
            redshift = 0.0,
            first_light = 5.0,
            fit_companion = False,
            lc_interp_array = [])
        np.savez('noise{:.2f}_sim__dynesty.npz'.format(noise_level), result)

    print(result['logwt'])
    PL_result = PL_Result( ['t_explosion',
                            'norm',
                            'index1',
                            'index2',
                            'baseline'],
                           result,)
    PL_result.params['redshift'] = ResultParam(0,0,0,0)
    PL_results.append(PL_result)
    
        #print(fit_result.params)

    plot_params = {}
    #for k in fit_result.params.keys():
    for k in PL_result.params.keys():
        #plot_params[k] = fit_result.params[k].value
        plot_params[k] = PL_result.params[k].value
    #print(plot_params)
    plot_y = curved_powerlaw_with_baseline(x,
                                           plot_params,
                                           )
    axes[ii+1].plot(x,plot_y,'r')

    if ii < 6:
        axes[ii+1].text(8,0,'noise level = {:.2f}; fitted index = {:.2f}'.format(noise_level,PL_result.params['index1'].value))
        axes[ii+1].yaxis.set_major_locator(MultipleLocator(1))
    elif ii == 6:
        axes[ii+1].text(8,-0.5,'noise level = {:.2f}; fitted index = {:.2f}'.format(noise_level,PL_result.params['index1'].value))
        axes[ii+1].yaxis.set_major_locator(MultipleLocator(1))

    elif ii == len(noise_levels)-1:
        axes[ii + 1].set_ylim([-6.5, 6.5])
        axes[ii + 1].text(8,-5.5,'noise level = {:.2f}; fitted index = {:.2f}'.format(noise_level,PL_result.params['index1'].value))
        axes[ii+1].yaxis.set_major_locator(MultipleLocator(3))

    else:
        axes[ii+1].set_ylim([-3.5, 3.5])
        axes[ii+1].text(8,-3,'noise level = {:.2f}; fitted index = {:.2f}'.format(noise_level,PL_result.params['index1'].value))
        axes[ii+1].yaxis.set_major_locator(MultipleLocator(2))

    l,h = axes[ii+1].get_ylim()
    print('look here',plot_params['t_explosion'],l,h)
    axes[ii+1].plot([plot_params['t_explosion'],
                     plot_params['t_explosion']],
                    [l,h],'r--',zorder=3)
    axes[ii+1].plot([0,0],[l,h],'k--')

        
pl_indices = np.array([res.params['index1'].value for res in PL_results])
t_exps = np.array([res.params['t_explosion'].value for res in PL_results])
epl_indices = np.array([res.params['index1'].stderr for res in PL_results])/4
et_exps = np.array([res.params['t_explosion'].stderr for res in PL_results])/4
print(np.c_[noise_levels, t_exps, pl_indices])

ax2.errorbar(t_exps,pl_indices,
                      epl_indices,et_exps,
                      fmt='k.',zorder=0)
colors = ax2.scatter(t_exps,
                     pl_indices,
                     c=np.log10(noise_levels),
                     s=64,
                     cmap='PuBu_r')
plt.colorbar(colors,label='log Noise [Fraction of Peak]')

plt.figtext(0.05,0.40,'Flux (Arbitrary)',rotation=90,fontsize=22)
axes[-1].set_xlabel("Time from first light (days)")

ax2.set_xlabel('Fitted Time of First Light $t_0$ (days)')
ax2.set_ylabel('Fitted PL Index $\\beta_1$')
l,h = ax2.get_xlim()
ax2.plot([l,h],[2,2],'k--')
ax2.set_xlim([l,h])

l,h = ax2.get_ylim()
ax2.plot([0,0],[l,h],'k--')
ax2.set_ylim([l,h])

F.subplots_adjust(hspace=0)
F.set_size_inches(12,8)
F.savefig('simulated_detection_thresh.png')
#plt.show()
