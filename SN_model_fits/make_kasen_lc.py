import numpy as np
import scipy as sp
from kasen_2010 import *
from astropy.table import Table


#redshift for 100 Mpc: 0.02295
redshift = 0.02295

R_SUN = 6.95501e10 #cm                                                                    
                                                                                          
#separations                                                                              
a1 = 1.008 #times 10^13 cm --> 50 R_sun                                                   
a2 = 0.5004*5./4.96 #times 10^13 cm --> 25 R_sun
a3 = 0.2500*10.0/12.4 #times 10^13 cm --> 10 R_sun
a4 = 0.1 #times 10^13 cm --> 5 R_sun                                                      
a5 = 0.1/4.96 #times 10^13 cm --> 1 R_sun                                                 
a6 = 0.05/4.96 #times 10^13 cm --> 0.5 R_sun
a7 = 0.01/4.96 #times 10^13 cm --> 0.1 R_sun

q = 1.0/1.4 #1 M_sun companion, 1.4 M_sun white dwarf                                     
r1 = roche_lobe_radius(q,a1)*1.e13/R_SUN                                                  
r2 = roche_lobe_radius(q,a2)*1.e13/R_SUN                                                  
r3 = roche_lobe_radius(q,a3)*1.e13/R_SUN                                                  
r4 = roche_lobe_radius(q,a4)*1.e13/R_SUN
r5 = roche_lobe_radius(q,a5)*1.e13/R_SUN
r6 = roche_lobe_radius(q,a6)*1.e13/R_SUN
r7 = roche_lobe_radius(q,a7)*1.e13/R_SUN   

                                                                                          
sn1 = TESS_Observation('tess_angstroms.txt')
lc_out = []
for a in [a1,a2,a3,a4,a5,a6,a7]:

    t,lc = sn1.make_fine_light_curve(a,redshift)        
    if a == a1:
        lc_out.append(t)
    mag = -2.5*sp.log10(lc) + 20.44
    lc_out.append(lc)
    lc_out.append(mag)

print(r1,r2,r3,r4,r5,r6,r7)
t = Table(lc_out,
          names=('time',
                 'e/s_{:04.1f}'.format(r1),
                 'mag_{:04.1f}'.format(r1),
                 'e/s_{:04.1f}'.format(r2),
                 'mag_{:04.1f}'.format(r2),
                 'e/s_{:04.1f}'.format(r3),
                 'mag_{:04.1f}'.format(r3),
                 'e/s_{:04.1f}'.format(r4),
                 'mag_{:04.1f}'.format(r4),
                 'e/s_{:04.1f}'.format(r5),
                 'mag_{:04.1f}'.format(r5),
                 'e/s_{:04.1f}'.format(r6),
                 'mag_{:04.1f}'.format(r6),
                 'e/s_{:04.1f}'.format(r7),
                 'mag_{:04.1f}'.format(r7),
          ))

print(np.array(lc_out)[:,0])

for c in t.colnames:
    if 'mag' in c:
        t[c].format = '%10.4f'
    elif 'e' in c:
        t[c].format = '%10.6f'
    else:
        t[c].format = '%10.6f'

t.write('kasen_lc_data.txt',format='ascii.csv',overwrite=True)
        
