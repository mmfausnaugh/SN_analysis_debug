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

#models from Kasen 2010, calcualted Teff, do BB spectrum, synphot, and
#r_p, and return in flux

class TESS_Observation:
    effective_area = 90.09 #cm^2
    def __init__(self,bandpass):
        print(bandpass)
        if os.path.isfile(bandpass):
            self.bandpass = S.FileBandpass(bandpass)
        else:
            print(bandpass)
            self.bandpass = S.ObsBandpass(bandpass)



    def synphot(self, Teff, r, z, A_V):
        #assume r is in cm (radius of emmitting region
        distance_kpc = lumdist(z)*1.e3 #uncertainty on z

        #default is photons/s/cm^2/angstrom,
        #pysynphot scales to solid angle omega_0 = sp.pi*R_sun**2/1kpc**2
        bb = S.BlackBody(Teff).redshift(z)
        #extinguish the spectrum here
        #pysynphot needs E(B-V), which is A_V/R_V
        ext = S.Extinction(A_V/3.1)
        obs = S.Observation(bb*ext, self.bandpass)
        #rescale to distance/surface area
        #flux = flux*omega_1/omega_0 = flux*(A/sp.pi/R_sun**2)*(1kpc/D**2)
        solid_angle = (r**2/R_SUN**2)/(distance_kpc)**2
        
        #photons/s/cm^2
        flux = obs.integrate()*solid_angle
        #return photons/s (electrons per second)
        #uncertainty on TESS solid angle
        return flux*self.effective_area



    def make_light_curve(self, separation, z):
        times = sp.r_[1.e-3:16:100j]

        temperatures = Teff(times, separation)
        luminosities = L_bol(times, separation)
        r_photospheres = radius2(luminosities, temperatures)

        #print(r_photospheres.max()/R_SUN)
        lc_electrons_per_second = sp.array(
            [self.synphot(var[0], var[1], z) for var in zip(temperatures, r_photospheres)] 
        )
        return times*(1. + z) , lc_electrons_per_second

    
    def make_fine_light_curve(self, separation, z):
        #spaced about 30 minutes
        times = sp.r_[1.e-3:16:768j]

        temperatures = Teff(times, separation)
        luminosities = L_bol(times, separation)
        r_photospheres = radius2(luminosities, temperatures)

        #print(r_photospheres.max()/R_SUN)
        lc_electrons_per_second = sp.array(
            [self.synphot(var[0], var[1], z) for var in zip(temperatures, r_photospheres)] 
        )
        return times*(1.+z),lc_electrons_per_second

    def make_fine_long_light_curve(self, separation, z, A_V = 0):
        #lasts  days, space by .042 days (about 1hour),
        #in order to compre with real data
        #times = sp.r_[1.e-3:50:0.042]
        times = sp.r_[1.e-3:50:200j]

        temperatures = Teff(times, separation)
        luminosities = L_bol(times, separation)
        r_photospheres = radius2(luminosities, temperatures)

        #print(r_photospheres.max()/R_SUN)
        lc_electrons_per_second = sp.array(
            [self.synphot(var[0], var[1], z, A_V) for var in zip(temperatures, r_photospheres)] 
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


    def make_timeseries_SED(self, t0, t1, Nsamples, separation, z):
        times = sp.r_[t0 : t1 : 1j*Nsamples]

        temperatures = Teff(times, separation)
        luminosities = L_bol(times, separation)
        r_photospheres = radius2(luminosities, temperatures)

        print(r_photospheres.max()/R_SUN)
        outwv,outflux = [],[]
        outbbwv,outbbflux = [],[]
        temp_out = []
        print('kasen temps',temperatures)
        print('kasen r_photospheres',r_photospheres)
        for var in zip(temperatures, r_photospheres):
            print('kasen',var)
            wv1,flux1,wv2,flux2,temp1 = self.get_SED(var[0], var[1], z)
            outwv.append(wv1)
            outflux.append(flux1)
            outbbwv.append(wv2)
            outbbflux.append(flux2)
            temp_out.append(temp1)
        return times*(1. + z) , outwv, outflux, outbbwv,outbbflux,temp_out

    def get_timeseries_teff_and_radius(self, t0, t1, Nsamples, separation, z):
        times = sp.r_[t0 : t1 : 1j*Nsamples]
        
        temperatures = Teff(times, separation)
        luminosities = L_bol(times, separation)
        r_photospheres = radius2(luminosities, temperatures)
        return temperatures, r_photospheres

#Kasen equations    
def Teff(t,a):
    #equation 25
    #time is in days, separation a is in 1.e13 cm
    #ignoring opacity here, but scales as (kappa/thomson)**(-35/36)

    #Kelvin
    return 2.5e4*a**0.25*t**(-37./72.)


#this one is a litte more complicated, whereas I can get it from stephan boltzmann law
##def radius(t,a):
##    #r_p in equation 24, radius of photosphere (of emmitting region)
##    n = 10
##
##    a = t**( (n-3)/(n-1))
##    b = (kappa_electron*)
##    c = b**(1./n-1)
##    return 
    
def L_bol(t,a):
    #equation 22 t is in days, a is in 1.e13 cm

    #assumes an ejecta velocity of 1.e9 cm/s, thomson oppacity 0.2 cm^2/g,
    #M_ejecta = M_Chandrasekhar = 1.4 M_sun, and a viewing angle of
    #45 degrees

    #erg per second
    return 1.e43*a*t**(-0.5)

def radius2(L,Teff):
    #return implied radius from stephan-boltzmann law
    return sp.sqrt( L/4/sp.pi/Teff**4/sigma)

def roche_lobe_radius(q,a):
    #q is mass ratio, a is separation
    f1 = 0.38*0.2*sp.log(q)
    f2 = 0.46224*(q/(1+q))**(1./3)

    return max(f1,f2)*a

