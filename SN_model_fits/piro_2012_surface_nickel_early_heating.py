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



    def make_light_curve(self, X_nickel, z):
        times = sp.r_[1.e-3:14:100j]

        temperatures = Teff(times,X_nickel)
        #luminosities = L_bol(times, X_nickel)
        #r_photospheres = radius2(luminosities, temperatures)
        r_photospheres = radius(times)

        #print(r_photospheres.max()/R_SUN)
        lc_electrons_per_second = sp.array(
            [self.synphot(var[0], var[1], z) for var in zip(temperatures, r_photospheres)] 
        )
        return times*(1. + z) , lc_electrons_per_second

    
    def make_fine_light_curve(self, z):
        #spaced about 30 minutes
        times = sp.r_[1.e-3:14:667j]

        temperatures = Teff(times)
        #luminosities = L_bol(times)
        #r_photospheres = radius2(luminosities, temperatures)
        r_photospheres = radius(times)

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


    def make_timeseries_SED(self, t0,t1,Nsamples, X_nickel, z):
        times = sp.r_[t0 : t1 : 1j*Nsamples]

        print('piro times',times)

        temperatures = Teff(times, X_nickel)
        #luminosities = L_bol(times, X_nickel)
        #r_photospheres = radius2(luminosities, temperatures)
        r_photospheres = radius(times)
        
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

    def get_timeseries_teff_and_radius(self, t0, t1, Nsamples, X_nickel, z):
        times = sp.r_[t0 : t1 : 1j*Nsamples]

        temperatures = Teff(times, X_nickel)
        #luminosities = L_bol(times, X_nickel)
        #r_photospheres = radius2(luminosities, temperatures)
        r_photospheres = radius(times)
        return temperatures, r_photospheres


#Piro equations

def Teff(t,X_nickel):
    #equation 23
    #1.6e4 Kelvin * X_56^*(0.25) * kappa_0.2^(-0.28) * E_51^(-0.11) * M_1.4^(0.1) * R_8.5^(0.03) * t_day^(0.05)
    #time is in days,
    #X_56 = mass fraction of nickel
    #kappa_0.2 = 0.5*kappa_thomson = 0.2 cm^2/g
    #(for some reason, Piro scaled the oppacity to 0.2 cm^2/g)
    #E_51 = 10^51 erg
    #M=  = 1.4 solar masses
    #R = 3.e8 cm, I think this is the initial size of the white dwarf ?

    #Kelvin
    return 1.6e4*X_nickel**(0.25)*t**(0.05)*2**(-0.28)


def radius(t):
    #photospheric in equation 25
    #2.2e14 cm * kappa_0.2^(0.1) * E_51^(0.45) * M_1.4^(-0.38) * R_8.5^(0.1) * t_day^(0.8)
    #time is in days,
    #kappa_0.2 = 0.5*kappa_thomson = 0.2 cm^2/g
    #(for some reason, Piro scaled the oppacity to 0.2 cm^2/g)
    #E_51 = 10^51 erg
    #M=  = 1.4 solar masses
    #R = 3.e8 cm, I think this is the initial size of the white dwarf ?

    #cm
    return 2.2e14*t**0.8*2**0.1

def L_bol(t,X_nickel):
    #equation 20
    #2.7e42 erg/s * X_56 * kappa_0.2^(-1) * E_51^(0.5) * M_1.4^(-0.5) t_day^(2)
    #time is in days,
    #X_56 = mass fraction of nickel
    #kappa_0.2 = 0.5*kappa_thomson = 0.2 cm^2/g
    #(for some reason, Piro scaled the oppacity to 0.2 cm^2/g)
    #E_51 = 10^51 erg
    #M=  = 1.4 solar masses
    #R = 3.e8 cm, I think this is the initial size of the white dwarf ?

    #erg per second
    return 2.7e42*X_nickel*t**(2)

def radius2(L,Teff):
    #return implied radius from stephan-boltzmann law
    return sp.sqrt( L/4/sp.pi/Teff**4/sigma)

