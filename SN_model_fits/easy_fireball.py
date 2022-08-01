#the idea here is to just put in plausible Teff and radius and get synthetic photometry

import scipy as sp
import matplotlib.pyplot as plt
import pysynphot as S
import os
from scipy.integrate import quad

#from tools.tools import lumdist
def lumdist(z):
    def E(z,Omega_m):
        #assumes flat universe
        return 1./sp.sqrt(
            Omega_m*(1.+z)**3 + (1. - Omega_m)
        )
    H0 = 70
    Omega_m = 0.3
    c  = 2.99792458e10

    #calculate the luminosity distnace, returns in Mpc
    integral = quad(E, 0, z, args=(Omega_m))
    #h0 in km/s
    dp = c/(H0 /3.08568e13/1.e6)*integral[0]
    
    return (1+z)*dp/3.08568e18/1.e6

kappa_electron = 0.2 #cm^2/g
c = 2.99792458e10 #cm/s
sigma = 5.6704e-5

CM_PER_PARSEC = 3.086e18
R_SUN = 6.95508e10

#models from Piro 2010, calcualted Teff, do BB spectrum, synphot, and
#r_p, and return in flux
#Teff is his equation 23
#rphot is his equ 25, which relies on the equation for luminosity eq. 20

class TESS_Observation:
    effective_area = 90.09 #cm^2
    def __init__(self,bandpass):
        print(bandpass)
        if os.path.isfile(bandpass):
            self.bandpass = S.FileBandpass(bandpass)
        else:
            print(bandpass)
            self.bandpass = S.ObsBandpass(bandpass)



    def synphot(self, Teff, r, z):
        #assume r is in cm (radius of emmitting region
        distance_kpc = lumdist(z)*1.e3
        #default is photons/s/cm^2/angstrom,
        #scaled to solid angle omega_0 = sp.pi*R_sun**2/1kpc**2
        bb = S.BlackBody(Teff).redshift(z)
        obs = S.Observation(bb, self.bandpass)
        #rescale to distance/surface area
        #flux = flux*omega_1/omega_0 = flux*(A/sp.pi/R_sun**2)*(1kpc/D**2)
        solid_angle = (r**2/R_SUN**2)/(distance_kpc)**2
        
        #photons/s/cm^2
        flux = obs.integrate()*solid_angle
        #return photons/s (electrons per second)
        return flux*self.effective_area



    def make_light_curve(self, Teff_use,v0, z):
        times = sp.r_[1.e-3:16:100j]

        temperatures = Teff(times,Teff_use)
        #luminosities = L_bol(times, X_nickel)
        #r_photospheres = radius2(luminosities, temperatures)
        r_photospheres = radius(times,v0)

        #print(r_photospheres.max()/R_SUN)
        lc_electrons_per_second = sp.array(
            [self.synphot(var[0], var[1], z) for var in zip(temperatures, r_photospheres)] 
        )
        return times*(1. + z) , lc_electrons_per_second

    
    def make_fine_light_curve(self, Teff_use,v0,z):
        #spaced about 30 minutes
        times = sp.r_[1.e-3:16:768j]

        temperatures = Teff(times,Teff_use)
        #luminosities = L_bol(times)
        #r_photospheres = radius2(luminosities, temperatures)
        r_photospheres = radius(times,v0)

        #print(r_photospheres.max()/R_SUN)
        lc_electrons_per_second = sp.array(
            [self.synphot(var[0], var[1], z) for var in zip(temperatures, r_photospheres)] 
        )
        return times*(1.+z),lc_electrons_per_second

    def get_SED(self,Teff,r,z):
        #assume r is in cm (radius of emmitting region
        distance_kpc = lumdist(z)*1.e3
        #default is photons/s/cm^2/angstrom,
        #scaled to solid angle omega_0 = sp.pi*R_sun**2/1kpc**2
        bb = S.BlackBody(Teff).redshift(z)
        obs = S.Observation(bb, self.bandpass)
        #rescale to distance/surface area
        #flux = flux*omega_1/omega_0 = flux*(A/sp.pi/R_sun**2)*(1kpc/D**2)
        solid_angle = (r**2/R_SUN**2)/(distance_kpc)**2
        
        #photons/s/cm^2
        flux = obs.flux*solid_angle
        #return photons/s (electrons per second)
        s_idx1 = bb.name.find('=')
        s_idx2 = bb.name.find(')')
        return obs.wave, flux,bb.wave,bb.flux*solid_angle,bb.name[s_idx1 + 1 :s_idx2]


    def make_timeseries_SED(self, t0,t1,Nsamples, Teff_use, v0, z):
        times = sp.r_[t0 : t1 : 1j*Nsamples]

        print('piro times',times)

        temperatures = Teff(times, Teff_use)
        #luminosities = L_bol(times, X_nickel)
        #r_photospheres = radius2(luminosities, temperatures)
        r_photospheres = radius(times,v0)
        
        print(r_photospheres.max()/R_SUN)
        outwv,outflux = [],[]
        outbbwv,outbbflux = [],[]
        temp_out = []
        print('piro temps',temperatures)
        print('piro r_photospheres',r_photospheres)
        for var in zip(temperatures, r_photospheres):
            print('piro',var)
            wv1,flux1,wv2,flux2,temp1 = self.get_SED(var[0], var[1], z)
            outwv.append(wv1)
            outflux.append(flux1)
            outbbwv.append(wv2)
            outbbflux.append(flux2)
            temp_out.append(temp1)
        return times*(1. + z) , outwv, outflux, outbbwv,outbbflux,temp_out

    def get_timeseries_teff_and_radius(self, t0, t1, Nsamples, Teff_use,v0, z):
        times = sp.r_[t0 : t1 : 1j*Nsamples]

        temperatures = Teff(times, Teff_use)
        #luminosities = L_bol(times, X_nickel)
        #r_photospheres = radius2(luminosities, temperatures)
        r_photospheres = radius(times, v0)
        return temperatures, r_photospheres


#make it a homologeously expanding fireball equations

def Teff(t,Teff_0):
    #for a fireball, this doesn't change
    return sp.array([Teff_0]*len(t))


def radius(t,v0):
    #constant velocity, radius is just v0*t
    #cm
    #initial radius is a typical white dwarf
    return v0*t*86400  + 3.0e8
