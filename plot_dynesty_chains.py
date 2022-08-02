import numpy as np
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot
import os
import sys
import glob
import re



ifiles = sys.argv[1:]
#ifiles = glob.glob('dynesty_chains/*')
for ifile in ifiles:

    try:
        obj = re.search('20(\d\d)(\w*)_',ifile)
        obj = '20' + obj.group(1) + obj.group(2)
    except AttributeError:
        obj=''
    print(obj)


    f = np.load(ifile,allow_pickle=True)
    results = f['arr_0'].tolist()
    print(results['logz'])
    weight = np.exp(results['logwt'] - results['logz'][-1])
    print(np.c_[weight[0:10],weight[-10:]])
    #print()
    labels = ['First Light $t_0$',
              'Normalization C',
              'PL Index $\\beta_1$',
              'PL Index $\\beta_2$',
              'Background $f_0$']

    ndim = np.shape(results['samples'])[1]
    if 'no_companion' in ifile:
        if ndim == 6:
            labels.append(['S2 Offset'])
    else:
        if ndim == 6:
            labels.append(['Companion Index'])
        elif ndim == 7:
            labels.append(['S2 Offset'])
            labels.append(['Companion Index'])
    
    fig,axes = dyplot.cornerplot(results,
                                 #show_titles=True,
                                 smooth=50, #defines Nbins
                                 labels=labels,
                                 show_titles=False)

    plt.gcf().suptitle('SN' + obj,fontsize=28)
    plt.savefig(ifile+'corner.png')
    fig,axes = dyplot.traceplot(results,
                                show_titles=True,
                                smooth=50,
                                labels=labels)
    plt.gcf().suptitle(ifile)
    plt.savefig(ifile+'trace.png')
    
    #plt.show()
