import scipy as sp
import matplotlib
matplotlib.rc('ytick',labelsize=14)
import matplotlib.pyplot as plt

from kasen_2010 import *

R_SUN = 6.95501e10 #cm

#separations
a1 = 1.008 #times 10^13 cm --> 50 R_sun
a2 = 0.1 #times 10^13 cm --> 5 R_sun
a3 = 0.1/4.96 #times 10^13 cm --> 1 R_sun
a4 = 0.01/4.96 #times 10^13 cm --> 0.1 R_sun

sn1 = TESS_Observation('tess_angstroms.txt')
t1,lc1 = sn1.make_light_curve(a1,0.05)
sn2 = TESS_Observation('tess_angstroms.txt')
t2,lc2 = sn1.make_light_curve(a2,0.05)
sn3 = TESS_Observation('tess_angstroms.txt')
t3,lc3 = sn1.make_light_curve(a3,0.05)


sn4 = TESS_Observation('tess_angstroms.txt')
t4,lc4 = sn1.make_light_curve(a1,0.01)
sn5 = TESS_Observation('tess_angstroms.txt')
t5,lc5 = sn1.make_light_curve(a2,0.01)
sn6 = TESS_Observation('tess_angstroms.txt')
t6,lc6 = sn1.make_light_curve(a3,0.01)


sn7 = TESS_Observation('tess_angstroms.txt')
t7,lc7 = sn1.make_light_curve(a4,0.05)
sn8 = TESS_Observation('tess_angstroms.txt')
t8,lc8 = sn1.make_light_curve(a4,0.01)

q = 1.0/1.4 #1 M_sun companion, 1.4 M_sun white dwarf
r1 = roche_lobe_radius(q,a1)*1.e13/R_SUN
r2 = roche_lobe_radius(q,a2)*1.e13/R_SUN
r3 = roche_lobe_radius(q,a3)*1.e13/R_SUN
r4 = roche_lobe_radius(q,a4)*1.e13/R_SUN

print('%e'%r1)
print('%e'%r2)
print('%e'%r3)

#F,(ax1,ax2)= plt.subplots(2,1,sharex='col')
F,(ax1)= plt.subplots(1,1,sharex='col')

ax1.plot(t4,lc4*1800*0.8*0.99,'k',label='${:3.1f}R_{{\\odot}}$, z = 0.01'.format(r1))
ax1.plot(t5,lc5*1800*0.8*0.99,'r',label='${:3.1f}R_{{\\odot}}$, z = 0.01'.format(r2))
ax1.plot(t6,lc6*1800*0.8*0.99,'b',label='${:3.1f}R_{{\\odot}}$, z = 0.01'.format(r3))
ax1.plot(t8,lc8*1800*0.8*0.99,'c',label='${:3.1f}R_{{\\odot}}$, z = 0.01'.format(r4))
ax1.plot(t1,lc1*1800*0.8*0.99,'k--',label='${:3.1f}R_{{\\odot}}$, z = 0.05'.format(r1))


ax1.text(11, 8.e4,'${:3.1f}R_{{\\odot}}$'.format(r1),fontsize='large')
ax1.text(11, 3.e4,'${:3.1f}R_{{\\odot}}$'.format(r2),fontsize='large')
ax1.text(11, 3.e3,'${:3.1f}R_{{\\odot}}$'.format(r3),fontsize='large')
ax1.text(11, 20,'${:3.1f}R_{{\\odot}}$'.format(r4),fontsize='large')

ax1.plot(t2,lc2*1800*0.8*0.99,'r--')
ax1.plot(t3,lc3*1800*0.8*0.99,'b--')
ax1.plot(t7,lc7*1800*0.8*0.99,'c--')



#ax2.plot(t1,-2.5*sp.log10(lc1) + 20.44,'k--',label='${:3.1f}R_{{\\odot}}$, z = 0.05'.format(r1))
#ax2.plot(t2,-2.5*sp.log10(lc2) + 20.44,'r--',label='${:3.1f}R_{{\\odot}}$, z = 0.05'.format(r2))
#ax2.plot(t3,-2.5*sp.log10(lc3) + 20.44,'b--',label='${:3.1f}R_{{\\odot}}$, z = 0.05'.format(r3))
#ax2.plot(t7,-2.5*sp.log10(lc7) + 20.44,'c--',label='${:3.1f}R_{{\\odot}}$, z = 0.05'.format(r3))
#
#
#ax2.plot(t4,-2.5*sp.log10(lc4) + 20.44,'k')
#ax2.plot(t5,-2.5*sp.log10(lc5) + 20.44,'r')
#ax2.plot(t6,-2.5*sp.log10(lc6) + 20.44,'b',label='${:3.1f}R_{{\\odot}}$, z = 0.01'.format(r3))
#ax2.plot(t6,-2.5*sp.log10(lc8) + 20.44,'c')

l,h =ax1.get_xlim()
ax1.plot([l,h],[2.170e3,2.170e3],'k:',label='Typical 8hr\n$3\\sigma$ detection')
#ax1.plot([l,h],[7.1e4,7.1e4],'k-.',label='Typical FFI sky')
#ax2.plot([l,h],-2.5*sp.log10([2.170e3/1800/0.8/0.99,2.170e3/1800/0.8/0.99]) + 20.44,'k:',label='Typical 8hr\n$3\\sigma$ detection')
#ax2.plot([l,h],-2.5*sp.log10([7.1e4/1800/0.8/0.99,7.1e4/1800/0.8/0.99]) + 20.44,'k-.',label='Typical FFI sky')
ax1.set_xlim([l,14.0])
#ax2.set_xlim([l,14.0])

#l,h = ax2.get_ylim()
ax1.set_ylim([10,3.e5])
#ax2.set_ylim([24,15])

ax1.set_yscale('log')
ax1.set_ylabel('electrons per FFI')
#ax2.set_ylabel('TESS mag',fontsize='x-large')
ax1.set_xlabel('Time since explosion (days)')
#F.subplots_adjust(hspace=0)

#ax2.set_xticklabels(ax1.get_xticks(),fontsize=14)
#ax1.set_yticklabels(ax1.get_yticks(),fontsize='large')
#ax2.set_yticklabels(ax2.get_yticks().astype(float),fontsize=14)

#ax1.legend(loc='lower right', fontsize='large')

#tick_labels = ax1.get_yticks()
#tick_labels = -2.5*sp.log10(tick_labels/1800/0.8/0.99) + 20.44
#print(tick_labels)
ax2 = ax1.twinx()
l,h = ax1.get_ylim()
l = -2.5*sp.log10(l/1800/0.8/0.99) + 20.44
h = -2.5*sp.log10(h/1800/0.8/0.99) + 20.44
print(l,h)
ax2.set_ylim([l,h])
ax2.set_ylabel('TESS mag')
F.set_size_inches(6,6)

F.savefig('kasen_2010_models_TESS_filter.pdf')

ax1.set_title('Kasen 2010 Models as TESS Observables')
F.savefig('kasen_2010_models_TESS_filter.png')
#plt.show()
