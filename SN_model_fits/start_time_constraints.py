import numpy as np
import matplotlib.pyplot as plt


#question: if TESS sees a delta mag above some limit, what can it
#actually say about the start time?

delta_mag = np.r_[0.1:5.5:50j]
flux_factor = 10**(0.4*delta_mag)
#assume constant temperature black body, what is the radius factor?
radius_factor = np.sqrt(flux_factor)
print(min(radius_factor),max(radius_factor))

F,(ax1,ax2) = plt.subplots(1,2)

ax1.plot(delta_mag,radius_factor)

#take a 1 R_sun source expanding at 10^4 km/s
#7.e5 #km, initional
#or a white dwarf,
R_init = 6.e3 
t = np.r_[0.0001:8:500j] #days
R_new = (t*86400*1.e4 + R_init)/R_init
flux_factor = R_new**2
mag = 2.5*np.log10(flux_factor)
ax2a = ax2.twinx()

ax2.plot(t,mag ,'k.-')
ax2a.plot(t,np.log10(R_new),'b.-')
#ax2.plot(t,flux_factor,'k')
l,h = ax2.get_ylim()
ax2.plot([1.0,1.0],[l,h],'k--')

#for a SN with a distance, a flux limit is a luminosity limit
#for a given temperture, a luminosity limit is a radius limit
#a radius limit is a time limit, for a given velocity and
#initial radius

plt.show()
