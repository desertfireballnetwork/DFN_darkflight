#!/usr/bin/env python
"""
Functions and objects to deal with meteoroids orbits
"""

__author__ = "Hadrien A.R. Devillepoix, Trent Jansen-Sturgeon "
__copyright__ = "Copyright 2016-2017, Desert Fireball Network"
__license__ = "MIT"
__version__ = "1.0"

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import HCRS, ITRS, GCRS
from astropy.utils.iers import IERS_A, IERS_A_URL, IERS
from astropy.utils.data import download_file

from trajectory_utilities import ECEF2LLH, \
     EarthPosition, HCRS2HCI, HCI2ECI_pos, \
     OrbitalElements2PosVel, ECI2ECEF_pos


try:
    iers_a_file = download_file(IERS_A_URL, cache=True)
    iers_a = IERS_A.open(iers_a_file)
    IERS.iers_table = iers_a
except:
    print('IERS_A_URL is temporarily unavailable')
    pass


AU = 1*u.au.to(u.m)
SMA_JUPITER = 5.20336301 * u.au



def tisserand_wrt_jupiter(a, e, i):
    '''
    Calculate the Tisserrand criterion with respect to Jupiter
    '''
    T_j = (SMA_JUPITER / a +
            2 * np.cos(i) *
            np.sqrt(a / SMA_JUPITER * (1 - e**2)))
    return T_j

# Conversion vector
AU_Deg2m_Rad = np.vstack((AU, 1, np.pi / 180 * np.ones((4, 1))))

Planets = {'Mercury': np.vstack((0.387099, 0.205636, 7.004979, 29.127030, 48.330766, 252.250324)),
           'Venus': np.vstack((0.723336, 0.006777, 3.394676, 54.922625, 76.679843, 181.979100)),
           'Earth': np.vstack((1.000003, 0.016711, -0.000015, 102.937682, 0.000000, 100.464572)),
           'Mars': np.vstack((1.523710, 0.093394, 1.849691, -73.503169, 49.559539, -4.553432)),
           'Jupiter': np.vstack((5.202887, 0.048386, 1.304397, -85.745429, 100.473909, 34.396441)),
           'Saturn': np.vstack((9.536676,0.053862,2.485992,-21.063546,113.662424,49.954244)),
           'Uranus': np.vstack((19.189165,0.047257,0.772638,96.937351,74.016925,313.238105)),
           'Neptune': np.vstack((30.069923,0.008590,1.770043,-86.819463,131.784226,-55.120030))}

class OrbitObject(object):
    """
    Solar system object osculating orbit
    """

    def __init__(self,
                 orbit_type,
                 a, e, i, omega, Omega, theta,
                 ra_corr=np.nan*u.rad, dec_corr=np.nan*u.rad,
                 v_g=np.nan*u.m/u.second):
        self.semi_major_axis = a.to(u.au)
        self.eccentricity = e
        self.inclination = i.to(u.deg)
        self.argument_periapsis = omega.to(u.deg)
        self.longitude_ascending_node = Omega.to(u.deg)
        self.longitude_perihelion = (self.longitude_ascending_node + self.argument_periapsis) % (360 * u.deg)

        self.true_anomaly = theta.to(u.deg)
        self.orbit_type = orbit_type

        self.perihelion = (1 - self.eccentricity) * self.semi_major_axis
        self.aphelion = (1 + self.eccentricity) * self.semi_major_axis

        self.corr_radiant_ra = (ra_corr.to(u.deg)) % (360 * u.deg)
        self.corr_radiant_dec = dec_corr.to(u.deg)

        radiant = HCRS(ra=self.corr_radiant_ra, dec=self.corr_radiant_dec, distance=1.0*u.au)
        ecpliptic_radiant = HCRS2HCI(np.vstack(radiant.cartesian.xyz.value))
        self.ecliptic_latitude = np.rad2deg(np.arcsin(ecpliptic_radiant[2] / norm(ecpliptic_radiant)))*u.deg

        self.velocity_g = v_g.to(u.m / u.second)

        self.T_j = self.tisserand_criterion_wrt_jupiter()

    def tisserand_criterion_wrt_jupiter(self):
        '''
        Calculate the Tisserrand criterion with respect to Jupiter
        '''
        return tisserand_wrt_jupiter(self.semi_major_axis, self.eccentricity, self.inclination)

    def __str__(self):
        return str("Semi-major axis:             " + str(self.semi_major_axis) + "\n" +
                   "Eccentricity:                " + str(self.eccentricity) + "\n" +
                   "Inclination:                 " + str(self.inclination) + "\n" +
                   "Argument of Periapsis:       " + str(self.argument_periapsis) + "\n" +
                   "Longitude of Ascending Node: " + str(self.longitude_ascending_node) + "\n" +
                   "True Anomaly:                " + str(self.true_anomaly) + "\n\n" +
                   "Ra_corrected:                " + str(self.corr_radiant_ra) + "\n" +
                   "Dec_corrected:               " + str(self.corr_radiant_dec) + "\n" +
                   "Vel_g:                       " + str(self.velocity_g))


'''
Function delibaretely outside of native StateVector class to allow multithreaded call
'''

def random_compute_orbit_ceplecha(sv):
    sv.randomize_velocity_vector()
    sv.computeOrbit(orbit_computation_method='Ceplecha')
    return sv

def random_compute_orbit_integration_EOE(sv):
    sv.randomize_velocity_vector()
    sv.computeOrbit(orbit_computation_method='integrate_EOE')
    return sv

def random_compute_orbit_integration_posvel(sv):
    sv.randomize_velocity_vector()
    sv.computeOrbit(orbit_computation_method='integrate_posvel')
    return sv


def PlotOrbitalElements(COE, t_jd, t_soi, Sol):

    Colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    i = 2 #FIXME error

    plt.figure()
    plt.subplot(321)
    plt.plot(t_jd, COE[0] / AU, Colour[i])
    plt.axvline(x=t_soi[0], color='b'); plt.grid()
    plt.xlabel("Time (JD)"); plt.ylabel("Semi-major Axis (AU)")
#    plt.axvline(x=t_soi[1], color='k')
#    plt.axvline(x=t_soi[2], color='c')


    plt.subplot(322)
    plt.plot(t_jd, COE[1], Colour[i])
    plt.axvline(x=t_soi[0], color='b'); plt.grid()
    plt.xlabel("Time (JD)"); plt.ylabel("Eccentricity")
#    plt.axvline(x=t_soi[1], color='k')
#    plt.axvline(x=t_soi[2], color='c')

    plt.subplot(323)
    plt.plot(t_jd, COE[2] * 180 / np.pi, Colour[i])
    plt.axvline(x=t_soi[0], color='b'); plt.grid()
    plt.xlabel("Time (JD)"); plt.ylabel("Inclination (deg)")
#    plt.axvline(x=t_soi[1], color='k')
#    plt.axvline(x=t_soi[2], color='c')

    plt.subplot(324)
    plt.plot(t_jd, COE[3] * 180 / np.pi, Colour[i])
    plt.axvline(x=t_soi[0], color='b'); plt.grid()
    plt.xlabel("Time (JD)"); plt.ylabel("Argument of Periapsis (deg)")
#    plt.axvline(x=t_soi[1], color='k')
#    plt.axvline(x=t_soi[2], color='c')

    plt.subplot(325)
    plt.plot(t_jd, COE[4] * 180 / np.pi, Colour[i])
    plt.axvline(x=t_soi[0], color='b'); plt.grid()
    plt.xlabel("Time (JD)"); plt.ylabel("Longitude of the Ascending Node (deg)")
#    plt.axvline(x=t_soi[1], color='k')
#    plt.axvline(x=t_soi[2], color='c')

    plt.subplot(326)
    plt.plot(t_jd, COE[5] * 180 / np.pi, Colour[i])
    plt.axvline(x=t_soi[0], color='b'); plt.grid()
    plt.xlabel("Time (JD)"); plt.ylabel("True Anomaly (deg)")
#    plt.axvline(x=t_soi[1], color='k')
#    plt.axvline(x=t_soi[2], color='c')

    if Sol != 'NoSol':
        plt.subplot(321)
        plt.axhline(Sol.semi_major_axis.value, color='g')
        plt.subplot(322)
        plt.axhline(Sol.eccentricity, color='g')
        plt.subplot(323)
        plt.axhline(Sol.inclination.value, color='g')
        plt.subplot(324)
        plt.axhline(Sol.argument_periapsis.value, color='g')
        plt.subplot(325)
        plt.axhline(Sol.longitude_ascending_node.value, color='g')
        plt.subplot(326)
        plt.axhline(Sol.true_anomaly.value, color='g')

    plt.show()


def PlotOrbit3D(OrbObjList, t0=2457535.0, Sol='NoSol'):

    from mpl_toolkits.mplot3d import Axes3D

    ''' 3D Orbit Plot'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for OrbObj in OrbObjList:
        COE = np.vstack((OrbObj.semi_major_axis.value,
                         OrbObj.eccentricity,
                         OrbObj.inclination.value,
                         OrbObj.argument_periapsis.value,
                         OrbObj.longitude_ascending_node.value,
                         OrbObj.true_anomaly.value)) * AU_Deg2m_Rad
        COE = COE + np.vstack((np.zeros((5, 100)), np.linspace(0, 2 * np.pi, 100)))
        [Pos_HCI, Vel_HCI] = OrbitalElements2PosVel(COE, 'Sun', 'Classical')
        ax.plot(Pos_HCI[0]/AU, Pos_HCI[1]/AU, Pos_HCI[2]/AU, color='r', label='Determined Orbit')

    ''' Plot the planets'''
    for Planet in Planets:
        COE = Planets[Planet] * AU_Deg2m_Rad
        COEs = COE + np.vstack((np.zeros((5, 200)), np.linspace(0, 2 * np.pi, 200)))
        [pos, vel] = OrbitalElements2PosVel(COEs, 'Sun', 'Classical')
        ax.plot(pos[0]/AU, pos[1]/AU, pos[2]/AU, color='b')

    # t_yr = t0 + np.linspace(0, 365.25, 100)
    # pos_earth = EarthPosition(t_yr)
    # ax.plot(pos_earth[0]/AU, pos_earth[1]/AU, pos_earth[2]/AU,
    #     color='b', linewidth=2.0, label='Earth')

    ''' Plot the solution (if given) '''
    if Sol != 'NoSol':
        Sol_oe = np.vstack((Sol.semi_major_axis.value,
                            Sol.eccentricity,
                            Sol.inclination.value,
                            Sol.argument_periapsis.value,
                            Sol.longitude_ascending_node.value,
                            Sol.true_anomaly.value)) * AU_Deg2m_Rad
        Sol_oe = Sol_oe + np.vstack((np.zeros((5, 100)), np.linspace(0, 2 * np.pi, 100)))
        [pos, vel] = OrbitalElements2PosVel(Sol_oe, 'Sun', 'Classical')
        ax.plot(pos[0]/AU, pos[1]/AU, pos[2]/AU, color='g', label='Published Orbit')

    plt.legend()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.show()

def PlotPerts(Pert):
    
    PPert = np.vstack(Pert).T; t = PPert[0]
    
    plt.figure(figsize=(16,9))
    t_rel = t - np.max(t) # Days
    plt.plot(t_rel, PPert[1], '-b', linewidth=3.0, label='Earth')
    plt.plot(t_rel, PPert[2], '--k', linewidth=3.0, label='Moon')
    plt.plot(t_rel, PPert[3], '-.r', linewidth=3.0, label='Sun')
    PertJ2 = PPert[4][~np.isnan(PPert[4])]
    plt.plot(t_rel[~np.isnan(PPert[4])], PertJ2, ':g', linewidth=3.0, label='J2')
    PertDrag = PPert[5][~np.isnan(PPert[5])]
    plt.plot(t_rel[~np.isnan(PPert[5])], PertDrag, '-.c', linewidth=3.0, label='Drag')
    plt.yscale('log'); plt.grid(True); plt.legend(loc='best')
    plt.xlabel('Relative Time [days]'); plt.ylabel('Perturbation Acceleration [m/s^2]')
    
    plt.show()

def PlotIntStep(t):
    
    dt=[]
    for k in range(len(t)-1):
        dt.append((t[k+1] - t[k]) * 24*60*60)
        
    plt.figure(figsize=(16,9))
    t_rel = t - np.max(t) # Days
    plt.plot(t_rel[1:], abs(np.array(dt)))
    plt.yscale('log'); plt.grid(True)#; plt.legend()
    plt.xlabel('Relative Time [days]'); plt.ylabel('Timestep [sec]')
    
    plt.show()


def ThirdBodyPerturbation(Pos, rho, mu):
    '''
    Pos is the position of the meteoroid (m)
    rho is the position of the third body (m)
    mu is the standard gravitational parameter of the third body (m3/s2)
    '''

    # Battin's scalar formula for vector difference
    q = np.dot(Pos.T, (Pos - 2 * rho) / (np.dot(rho.T, rho)))
    f = (3 * q + 3 * q**2 + q**3) / (1 + (1 + q)**1.5)

    # Third body perturbation acceleration (with indirect term)
    u = -mu * (Pos + f * rho) / ((norm(Pos - rho))**3)

    return u


def NRLMSISE_00(pos, time, pos_type='eci'):
    ''' Courtesy of Ellie Sansom '''
    """
    Inputs: inertial position and time
    Outputs: [altitude, temp, atm_pres, atm density, sos, dyn_vis]
    """

    from nrlmsise_00_header import nrlmsise_input, nrlmsise_output, nrlmsise_flags
    from nrlmsise_00 import gtd7

    time = Time(time, format='jd', scale='utc')

    # Convert ECI to LLH coordinates
    if pos_type == 'eci':
        Pos_LLH = ECEF2LLH(ECI2ECEF_pos(pos, time))
    elif pos_type == 'ecef':
        Pos_LLH = ECEF2LLH(pos)
    elif pos_type == 'llh':
        Pos_LLH = pos
    else:
        print('NRLMSISE_00 error: Invalid pos_type')
        exit()
    g_lat = np.rad2deg(Pos_LLH[0][0])
    g_long = np.rad2deg(Pos_LLH[1][0])
    alt = Pos_LLH[2][0]

    # Break up time into year, day of year, and seconds of the day
    yDay = time.yday.split(':'); yr = float(yDay[0]); doy = float(yDay[1])
    sec = float(yDay[2]) * 60*60 + float(yDay[3]) * 60 + float(yDay[4])

    # Assign our variables into the nrmsise inputs
    Input = nrlmsise_input(yr, doy, sec, alt/1000, g_lat, g_long)
    Output = nrlmsise_output(); Flags = nrlmsise_flags()

    # Switches
    for i in range(1, 24):
        Flags.switches[i]=1

    # GTD7 atmospheric model subroutine
    gtd7(Input, Flags, Output)

    # Temperature at alt [deg K]
    T = Output.t[1]

    # Molecular number densities [m-3]
    He = Output.d[0] # He
    O  = Output.d[1] # O
    N2 = Output.d[2] # N2
    O2 = Output.d[3] # O2
    Ar = Output.d[4] # Ar
    H  = Output.d[6] # H
    N  = Output.d[7] # N
#    ano_O  = Output.d[8] # Anomalous oxygen
    sum_mass = He + O + N2 + O2 + Ar + H + N

    # Molar mass
    He_mass = 4.0026  # g/mol
    O_mass  = 15.9994 # g/mol
    N2_mass = 28.013  # g/mol
    O2_mass = 31.998  # g/mol
    Ar_mass = 39.948  # g/mol
    H_mass  = 1.0079  # g/mol
    N_mass  = 14.0067 # g/mol

    # Molecular weight of air [kg/mol]
    mol_mass_air = (He_mass * He + O_mass * O + N2_mass * N2 + O2_mass * O2
                  + Ar_mass * Ar + H_mass * H + N_mass * N) / (1000 * sum_mass)

    # Total mass density [kg*m-3]
    po = Output.d[5] * 1000

    Ru = 8.3144621 # Universal gas constant [J/(K*mol)]
    R = Ru / mol_mass_air # Individual gas constant [J/(kg*K)] #287.058

    # Ideal gas law
    atm_pres = po * T * R

    # Speed of sound in atm
    sos = 331.3 * np.sqrt(1 + T / 273.15)

    # Dynamic viscosity (http://en.wikipedia.org/wiki/Viscosity)
    C = 120 #Sutherland's constant for air [deg K]
    mu_ref = 18.27e-6 # Reference viscosity [[mu_Pa s] * e-6]
    T_ref = 291.15 # Reference temperature [deg K]

    dyn_vis = mu_ref * (T_ref + C) / (T + C) * (T / T_ref)**1.5

    return T, atm_pres, po, sos, dyn_vis

# def compute_infinity_radiant(stateVec):
#     ''' This method computing the apparent radiant, it doesn't consider the zenith attraction '''

#     Pos_geo = stateVec.position
#     Vel_geo = stateVec.vel_xyz
#     t0 = stateVec.epoch

#     # Compute radiant (apparent ORIGIN of meteoroid)
#     Vel_eci = ECEF2ECI(Pos_geo, Vel_geo, t0)[1]
#     ra_eci = np.arctan2(-Vel_eci[1], -Vel_eci[0])
#     dec_eci = np.arcsin(-Vel_eci[2] / norm(Vel_eci))
#     # ^-- redundant information. Already have it in metadata

#     return ra_eci, dec_eci


def compute_cartesian_velocities_from_radiant(stateVec):
    '''
    Turn apparent ecef radiant and velocity into cartesian velocity component
    '''

    vel_geo = -(stateVec.velocity_inf *
               np.vstack((np.cos(np.deg2rad(stateVec.ra_ecef_inf)) * np.cos(np.deg2rad(stateVec.dec_ecef_inf)),
                          np.sin(np.deg2rad(stateVec.ra_ecef_inf)) * np.cos(np.deg2rad(stateVec.dec_ecef_inf)),
                          np.sin(np.deg2rad(stateVec.dec_ecef_inf)))))

    return vel_geo



def SimilarityCriterion(COE1, COE2, method='SH'):
    '''
    Southworth & Hawkins similarity criterion (1963); or
    Drummond's similarity criterion (1981); or
    Jopek's similarity criterion (1993).
    '''
    if type(COE1) == np.ndarray:
        a1 = COE1[0]/AU; a2 = COE2[0]/AU # [AU]
        e1 = COE1[1];    e2 = COE2[1]    # []
        i1 = COE1[2];    i2 = COE2[2]    # [rad]
        w1 = COE1[3];    w2 = COE2[3]    # [rad]
        W1 = COE1[4];    W2 = COE2[4]    # [rad]

    else:
        a1 = COE1.semi_major_axis.value;                    a2 = COE2.semi_major_axis.value                    # [AU]
        e1 = COE1.eccentricity;                             e2 = COE2.eccentricity                             # []
        i1 = COE1.inclination.to(u.rad).value;              i2 = COE2.inclination.to(u.rad).value              # [rad]
        w1 = COE1.argument_periapsis.to(u.rad).value;       w2 = COE2.argument_periapsis.to(u.rad).value       # [rad]
        W1 = COE1.longitude_ascending_node.to(u.rad).value; W2 = COE2.longitude_ascending_node.to(u.rad).value # [rad]

    q1 = a1 * (1 - e1) # [AU]
    q2 = a2 * (1 - e2) # [AU]

    # Angle between the orbital planes (I21)
    var = (2 * np.sin((i2 - i1) / 2))**2 + np.sin(i1) * np.sin(i2) * (2 * np.sin((W2 - W1) / 2))**2
    I21 = 2 * np.arcsin(np.sqrt(var) / 2)

    if method == 'SH':
        # Difference between orbits longitude of perihelion (pi21)
        pi21 = w2 - w1 + 2 * np.arcsin(np.cos((i2 + i1) / 2) * np.sin((W2 - W1) / 2) / np.cos(I21 / 2))

        Similarity2 = (e2 - e1)**2 + (q2 - q1)**2 + var + (((e2 + e1) / 2) * (2 * np.sin(pi21 / 2)))**2
        Similarity = np.sqrt(Similarity2)

    elif method == 'D':
        # Angle between the orbital lines of apsides (theta21)
#        l1 = W1 + np.arcsin(np.cos(i1) * np.tan(w1)); b1 =  np.arcsin(np.sin(i1) * np.sin(w1))
#        l2 = W2 + np.arcsin(np.cos(i2) * np.tan(w2)); b2 =  np.arcsin(np.sin(i2) * np.sin(w2))
        l1 = W1 + np.arctan(np.cos(i1) * np.tan(w1)); b1 =  np.arcsin(np.sin(i1) * np.sin(w1))
        l2 = W2 + np.arctan(np.cos(i2) * np.tan(w2)); b2 =  np.arcsin(np.sin(i2) * np.sin(w2))
        theta21 = np.arccos(np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l2 - l1))

        Similarity2 = ((e2 - e1) / (e2 + e1))**2 + ((q2 - q1) / (q2 + q1))**2 + \
                    (I21 / np.pi)**2 + ((e2 + e1) / 2)**2 * (theta21 / np.pi)**2
        Similarity = np.sqrt(Similarity2)

    elif method == 'H':
        # Difference between orbits longitude of perihelion (pi21)
        pi21 = w2 - w1 + 2 * np.arcsin(np.cos((i2 + i1) / 2) * np.sin((W2 - W1) / 2) / np.cos(I21 / 2))

        Similarity2 = (e2 - e1)**2 + ((q2 - q1) / (q2 + q1))**2 + var + \
                        (((e2 + e1) / 2) * (2 * np.sin(pi21 / 2)))**2
        Similarity = np.sqrt(Similarity2)

    return Similarity

def generate_ephemeris(pos_hci, t_jd):

    # Save the datetime
    ephem_dict = {'datetime': Time(t_jd, format='jd', scale='utc').isot}
    ephem_dict['MJD'] = Time(t_jd, format='jd', scale='utc').mjd
    
    # distance to sun
    ephem_dict['distance_to_sun'] = norm(pos_hci, axis=0) / 1000 #km

    # Convert to eci coordinates
    pos_eci = HCI2ECI_pos(pos_hci, t_jd)
    ephem_dict['pos_eci_x'] = pos_eci[0]
    ephem_dict['pos_eci_y'] = pos_eci[1]
    ephem_dict['pos_eci_z'] = pos_eci[2]
    pos_hcrs = HCI2HCRS(pos_hci)

    # Calculate phase angle
    ephem_dict['phase_angle'] = np.rad2deg(np.arccos(np.sum(pos_hcrs * pos_eci, axis=0)
        / (norm(pos_hcrs, axis=0) * norm(pos_eci, axis=0))))

    # Calculate elongation angle
    pos_sun = pos_eci - pos_hcrs
    ephem_dict['elongation_angle'] = np.rad2deg(np.arccos(np.sum(pos_sun * pos_eci, axis=0)
        / (norm(pos_sun, axis=0) * norm(pos_eci, axis=0))))

    # Calculate ephemeris
    dist = norm(pos_eci, axis=0) #m
    ephem_dict['ra'] = np.rad2deg(np.arctan2(pos_eci[1], pos_eci[0]))%360 #deg
    ephem_dict['dec'] = np.rad2deg(np.arcsin(pos_eci[2] / dist)) #deg
    ephem_dict['distance_to_earth'] = norm(pos_eci, axis=0) / 1000 #km

    return ephem_dict
