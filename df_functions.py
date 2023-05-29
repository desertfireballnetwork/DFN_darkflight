"""
Meteoroid Dark Flight Propagator

This dark flight model predicts the landing sight of a meteoroid by
propagating the position and velocity through the atmosphere using a
5th-order adaptive step size integrator (ODE45).

Created on Mon Oct 17 10:59:00 2016
@author: Trent Jansen-Sturgeon
"""
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d
from astropy.time import Time
from scipy.interpolate import griddata
from wrf import extract_times, getvar, ALL_TIMES

from trajectory_utilities import ECEF2LLH, \
    ECI2ECEF_pos, ECEF2ECI_pos, OMEGA_EARTH
from orbital_utilities import NRLMSISE_00

# def Propagate(data, WindData, dt, met_rho):
#     '''
#     Inputs: Initial ECI position (m), ECI velocity (m/s), and mass (kgs).
#     Outputs: ECI position (m), ECI velocity (m/s) throughout the dark flight.
#     '''

#     init_x = copy.deepcopy(data[1:11]) # Add mass to the state 
#     param = [WindData, met_rho]

#     ##integration:
#     ode_output = scipy.integrate.odeint(EarthDynamics, init_x, [0, dt], args = (param,)) 

#     ## set new particle
#     data[0] += dt       #float(str( Time(data[0], format='jd') + TimeDelta(dt, format='sec')))
#     data[1:11] = ode_output[1]
#     #data[12:] = 

#     return data

# def EarthDynamics(X, t, params):
#     '''
#     The state rate dynamics are used in Runge-Kutta integration method to 
#     calculate the next set of equinoctial element values.
#     ''' 
#     [WindData, met_rho] = params
#     mu_e = 3.986005000e14 #4418e14 # Earth's standard gravitational parameter (m3/s2)
#     w_e = 7.2921158553e-5  # Earth's rotation rate (rad/s)

#     ## State Rates 
#     # State parameter vector decomposed
#     Pos_ECI = np.vstack((X[:3]))
#     Vel_ECI = np.vstack((X[3:6]))
#     kappa_no_drag = X[7] / 1.3
#     A = kappa_no_drag * met_rho**(2./3)
#     M = X[6]
#     S = X[9]
#     sig = X[8]
#     mu = 2./3

#     ## Primary Gravitational Acceleration 
#     grav = - mu_e * Pos_ECI / norm(Pos_ECI)**3
    
#     ## Atmospheric Drag Perturbation - Better Model Needed '''
#     # Atmospheric velocity
#     [v_wind, rho_a, temp] = AtomosphericModel(WindData, Pos_ECI)
#     v_rot = np.cross(np.vstack((0, 0, w_e)), Pos_ECI, axis=0)
#     v_atm = v_rot + v_wind

#     # Velocity relative to the atmosphere
#     v_rel = Vel_ECI - v_atm
#     v = norm(v_rel, axis=0)

#     # Cross-sectional area <------- Assumption alert!!!
#     d = 2 * np.sqrt(S / np.pi) # Diameter of meteoroid (m)
    
#     # Constants for drag coeff
#     mu_a = viscosity(temp) # Air Viscosity (Pa.s)
#     mach = v / SoS(temp) # Mach Number
#     re = reynolds(rho_a, v, mu_a, A) # Reynolds Number
#     kn = knudsen(mach, re) # Knudsen Number
#     Cd = dragcoef(re, mach, kn, d) # Drag Coefficient
#     #Cd = 2.0 # Approximation

#     Xdot=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

#     kv = 0.5 * kappa_no_drag  * rho_a
#     km = kv * sig

#     Xdot[:3] = Vel_ECI.reshape(3)
#     Xdot[3:6] = (-kv * Cd * abs(M)**(mu-1) * v * v_rel + grav).reshape(3)
#     Xdot[6] = -km * Cd * v**3 * abs(M)**mu

#     return Xdot

def density_from_pressure(temperature, pressure, RH):
    """returns atmospheric density, (kg/m3)
    for a single point given:
    Pressure (Pascals, multiply mb by 100 to get Pascals)
    temperature ( deg K)
    RH (from 0 to 1 as fraction) """
    # R = specific gas constant , J/(kg*degK) = 287.05 for dry air
    Rd = 287.05
    # http://www.baranidesign.com/air-density/air-density.htm
    # http://wahiduddin.net/calc/density_altitude.htm
    # Evaporation into the Atmosphere, Wilfried Brutsaert, p37
    # saturation vapor pressure is a polynomial developed by Herman Wobus
    e_so = 6.1078
    c0 = 0.99999683
    c1 = -0.90826951e-2
    c2 = 0.78736169e-4
    c3 = -0.61117958e-6
    c4 = 0.43884187e-8
    c5 = -0.29883885e-10
    c6 = 0.21874425e-12
    c7 = -0.17892321e-14
    c8 = 0.11112018e-16
    c9 = -0.30994571e-19
    
    p = (c0 + temperature*(
         c1 + temperature*(
         c2 + temperature*(
         c3 + temperature*(
         c4 + temperature*(
         c5 + temperature*(
         c6 + temperature*(
         c7 + temperature*(
         c8 + temperature*(
         c9)))))))))) 
    
    sat_vp = e_so / p**8
    Pv = sat_vp * RH
    density = (pressure / (Rd * temperature)) * (1 - (0.378 * Pv / pressure))
    return density


################################################################
WRF_history = []
def AtmosphericModel(WindData, Pos_ECI, t_jd):

    # The rotational component of the wind in the ECI frame
    north_pole_ecef = np.vstack((0,0,6356.75231e3))
    north_pole_eci = ECEF2ECI_pos(north_pole_ecef, t_jd)
    earth_ang_rot = OMEGA_EARTH * north_pole_eci / norm(north_pole_eci)
    v_rot = np.cross(earth_ang_rot, Pos_ECI, axis=0)
    
    if len(WindData):

        Pos_LLH = ECEF2LLH(Pos_ECI) # There is a max of +-40m height error here
        h = float(Pos_LLH[2]) # (m)

        # Get the relevent wind data from that height
        if h > max(WindData['# Height']): # Think about using a model for +30km
            i = np.argmax(WindData['# Height'])
            Wind = WindData['Wind'][i] # (m/s)
            WDir = WindData['WDir'][i] # (deg)
            TempK = WindData['TempK'][i] # (deg K)
            Press = WindData['Press'][i] # (Pa)
            RHum = WindData['RHum'][i] # (%)
        
        elif h <= max(WindData['# Height']) and h >= min(WindData['# Height']):
            Wind = interp1d(WindData['# Height'], WindData['Wind'], kind='cubic')(h)
            WDir = interp1d(WindData['# Height'], WindData['WDir'], kind='cubic')(h)
            TempK = interp1d(WindData['# Height'], WindData['TempK'], kind='cubic')(h)
            Press = interp1d(WindData['# Height'], WindData['Press'], kind='cubic')(h)
            RHum = interp1d(WindData['# Height'], WindData['RHum'], kind='cubic')(h)
                
        elif h < min(WindData['# Height']):
            i = np.argmin(WindData['# Height'])
            Wind = WindData['Wind'][i] # (m/s)
            WDir = WindData['WDir'][i] # (deg)
            TempK = WindData['TempK'][i] # (deg K)
            Press = WindData['Press'][i] # (Pa)
            RHum = WindData['RHum'][i] # (%)
        else:
            print('There is a problem in AtomosphericModel.')
            print(Pos_ECI)
            print(min(WindData['# Height']))
            print(max(WindData['# Height']))
            exit()
        
        # Calculate the atmospheric density
        rho_a = density_from_pressure(TempK, Press, RHum) # (kg/m3)
        
        # Construct the wind vector (start with clockwise N=0, coming from!!)
        Wind_ENU = - np.vstack((Wind * np.sin(np.deg2rad(WDir)), 
                                Wind * np.cos(np.deg2rad(WDir)), 
                                0))
        
        # Convert the ENU to ECI coordinates (using lat/lon from ECI frame)
        lat = float(Pos_LLH[0]); lon = float(Pos_LLH[1]) # (rad)
        C_ENU2ECI = np.array([[-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                              [ np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                              [    0       ,         np.cos(lat)       ,        np.sin(lat)       ]])
        
        Wind_ECI = np.dot(C_ENU2ECI, Wind_ENU) + v_rot

    else:
        Pos_ECEF = ECI2ECEF_pos(Pos_ECI, t_jd)
        Pos_LLH = ECEF2LLH(Pos_ECEF)
        h = float(Pos_LLH[2]) # (m)

        [TempK, Press, rho_a] = NRLMSISE_00(Pos_LLH, t_jd, pos_type='llh')[:3]

        # Assume no wind
        Wind_ENU = np.vstack(( 0, 0, 0))
        Wind_ECI = v_rot

    # else:
    #     '''
    #     The atmospheric density using NASA's Earth Atmosphere Model.
    #     Input height above sea level (m).
    #     Outputs the atmospheric density (kg/m3) and the temperature (K) at the
    #     inputted height.
    #     Source: www.grc.nasa.gov/WWW/k-12/airplane/atmosmet.html
    #     '''
    
    #     # Upper Stratosphere
    #     if h >= 25000:  # (m)
    #         T = -131.21 + 0.00299 * h + 273.15  # (deg K)
    #         p = 2.488 * (T / 216.6)**(-11.388)  # (kPa)
    
    #     # Lower Stratosphere
    #     elif h >= 11000 and h < 25000:  # (m)
    #         T = -56.46 + 273.15  # (deg K)
    #         p = 22.65 * np.exp(1.73 - 0.000157 * h)  # (kPa)
    
    #     # Troposphere
    #     elif h >= 0 and h < 11000:  # (m)
    #         T = 15.04 - 0.00649 * h + 273.15  # (deg K)
    #         p = 101.29 * (T / 288.08)**(5.256)  # (kPa)
    
    #     elif h < 0:  # (m)
    #         T = 15.04 + 273.15  # (deg K)
    #         p = 101.29 * (T / 288.08)**(5.256)  # (kPa)
    
    #     else:
    #         print('Error: Consult ADM function')
    #         exit()
    
    #     # Air Density (kg/m3)
    #     rho_a = p / (0.2869 * T)
    #     TempK = T

    #     # Assume no wind
    #     Wind_ENU = np.vstack(( 0, 0, 0))
    #     Wind_ECEF = np.vstack(( 0, 0, 0))

    # Save the variables
    WRF_history.append( np.vstack((h, Wind_ENU, rho_a, TempK)) )

    return Wind_ECI, rho_a, TempK

from netCDF4 import Dataset
def WindDataExtraction(WindFileName, t0):
    # Interpolates temporally (with fixed lat/lon/pres?)
    from wrf import to_np

    if isinstance(t0, np.ndarray): t0 = t0[0]

    WindFile = Dataset(WindFileName)
    times_all = extract_times(WindFile, timeidx=ALL_TIMES)
    times_jd = np.array([Time(str(t), format='isot', scale='utc').jd for t in times_all])
    
    idx_after = np.searchsorted(times_jd, t0)
    idx_before = idx_after - 1

    interp_factor = (t0 - times_jd[idx_before]) / (times_jd[idx_after] - times_jd[idx_before])
    if interp_factor < 0 or interp_factor > 1:
        print('WindWarning: The darkflight time is ouside the bounds of WindData' \
            ' by {0:.3f} times!'.format(interp_factor))

    WindArray = []
    for i in [idx_before, idx_after]:
        hei_3d = np.array([to_np(getvar(WindFile,'z',timeidx=i))]) #[1,z,y,x]
        NumberLevels = np.shape(hei_3d)[1] # Number heights

        lat_3d = np.array([np.stack([to_np(getvar(WindFile,'lat',timeidx=i))]*NumberLevels, axis=0)]) #[1,z,y,x]
        lon_3d = np.array([np.stack([to_np(getvar(WindFile,'lon',timeidx=i))]*NumberLevels, axis=0)]) #[1,z,y,x]

        wen_3d = to_np(getvar(WindFile,'uvmet',timeidx=i)) #[2,z,y,x]
        wu_3d = np.array([to_np(getvar(WindFile,'wa',timeidx=i))]) #[1,z,y,x]

        temp_3d = np.array([to_np(getvar(WindFile,'tk',timeidx=i))]) #[1,z,y,x]
        pres_3d = np.array([to_np(getvar(WindFile,'p',timeidx=i))]) #[1,z,y,x]
        rh_3d = np.array([to_np(getvar(WindFile,'rh',timeidx=i))]) #[1,z,y,x]

        # Construct WindArray = [lat,lon,hei,we,wn,wu,temp,pres,rh]
        WindArray.append( np.vstack((lat_3d, lon_3d, hei_3d, wen_3d, 
                        wu_3d, temp_3d, pres_3d, rh_3d)) )

    WindArray = (1 - interp_factor) * WindArray[0] + interp_factor * WindArray[1]

    return WindArray

import warnings
from wrf import interplevel
def WRF3D(WindArray, Pos_LLH):
    # Interpolates spacially

    # Assign the lat/lon/hei and find height variation in the model
    [[lat],[lon],[hei]] = [np.rad2deg(Pos_LLH[0]), np.rad2deg(Pos_LLH[1]), Pos_LLH[2]]

    # Find xy positions of the lat/lon
    ang_dist2 = (WindArray[0,0] - lat)**2 + (WindArray[1,0] - lon)**2
    min_index = np.argmin(ang_dist2); ncol = WindArray.shape[3]
    xid = min_index % ncol; yid = min_index // ncol

    lat_var = WindArray[0,0,yid-1:yid+2,xid-1:xid+2] #[3,3]
    lon_var = WindArray[1,0,yid-1:yid+2,xid-1:xid+2] #[3,3]
    hei_var = WindArray[2,:,yid-1:yid+2,xid-1:xid+2] #[z,3,3]

    # Find the variable at a certain altitude as a 2D array [3,3] #linear interpolation!
    interp_horiz = lambda entry_no: interplevel(
        field3d=WindArray[entry_no,:,yid-1:yid+2,xid-1:xid+2], 
        vert=hei_var, desiredlev=hei, missing=np.nan).data #<---RuntimeWarning originates from here

    # 2D interpolate [1,]
    latlon = np.vstack((lat_var.flatten(), lon_var.flatten())).T
    interp2pt = lambda entry_no: griddata(latlon, 
        interp_horiz(entry_no).flatten(), np.array([lat,lon]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        we = interp2pt(3) # Wind east [m/s]
        wn = interp2pt(4) # Wind north [m/s]
        wu = interp2pt(5) # Wind up [m/s]
        tk = interp2pt(6) # Temperature [K]
        pr = interp2pt(7) # Pressure [Pa]
        rh = interp2pt(8) # Relative humidity []

    # Compare:
    rho_a = density_from_pressure(tk, pr, rh)
    if np.isnan(rho_a): [hei, we, wn, wu, rho_a, tk] = WRF_history[-1]
    Wind_ENU = np.vstack((we, wn, wu))

    # Save the variables
    WRF_history.append( np.vstack((hei, Wind_ENU, rho_a, tk)) )

    return Wind_ENU, rho_a, tk

# from wrf import getvar, interplevel, interpline, CoordPair
# def WRF_3D(data, Pos_LLH):

#     # Assign the lat/lon/hei and find height variation in the model
#     [lat,lon,hei] = [np.rad2deg(Pos_LLH[0]), np.rad2deg(Pos_LLH[1]), Pos_LLH[2]]
#     hei_var = getvar(data, 'z')

#     # Find the variable at a certain altitude as a 2D array
#     interp_horiz = lambda field, val: interplevel(field3d=getvar(data, field), 
#         vert=hei_var, desiredlev=hei, missing=np.nan)
#     wen_2D = interp_horiz('uvmet') # Wind east/north [m/s]
#     wu_2D = interp_horiz('wa') # Wind up [m/s]
#     tk_2D = interp_horiz('tk') # Temperature [K]
#     pr_2D = interp_horiz('p') # Pressure [Pa]
#     rh_2D = interp_horiz('rh') # Relative humidity []

#     # Stack the variables for interpolation [we, wn, wu, tempK, pres, RH]
#     variable_stack = np.vstack((wen_2D, np.dstack(wu_2D.T),np.dstack(tk_2D.T),
#                                 np.dstack(pr_2D.T), np.dstack(rh_2D.T)))
#     try:
#         [we, wn, wu, tempK, pres, RH] = interpline(variable_stack, wrfin=data, latlon=True,
#             start_point=CoordPair(lat=lat, lon=lon), end_point=CoordPair(lat=lat, lon=lon+0.1))[:,0]
#     except:
#         [we, wn, wu, tempK, pres, RH] = interpline(variable_stack, wrfin=data, latlon=True,
#             start_point=CoordPair(lat=lat, lon=lon), end_point=CoordPair(lat=lat, lon=lon-0.1))[:,0]

#     # Compare:
#     rho_a = density_from_pressure(tempK, pres, RH)
#     if np.isnan(rho_a): [we, wn, wu, rho_a, tempK] = WRF_history[-1]
#     Wind_ENU = np.vstack((we, wn, wu))

#     # Save the variables
#     WRF_history.append( np.vstack((Wind_ENU, rho_a, tempK)) )

#     return Wind_ENU, rho_a, tempK

