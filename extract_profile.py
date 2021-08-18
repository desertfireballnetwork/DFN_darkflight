"""
Extract a vertical atmospheric profile
from a WRF data product.
Units are all SI.
Angle in degrees [0, 360].
See argmument help by running with '-h' = '--help'

Install dependencies in a conda environment:
conda create --name wrf
conda activate wrf
conda install pip
pip install astropy pandas scipy numpy
pip install wrf-python netcdf4 
"""
__author__ = "Hadrien A R Devillepoix"
__license__ = "MIT"
__version__ = "1.0"

import os
import warnings

from netCDF4 import Dataset
from wrf import getvar
import wrf

from astropy.time import Time
import numpy as np
from numpy import linalg as LA
from scipy.interpolate import griddata
import pandas as pd


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


def WindDataExtraction(WindFileName, t0):
    # Interpolates temporally (with fixed lat/lon/pres?)
    from wrf import to_np

    if isinstance(t0, np.ndarray): t0 = t0[0]

    WindFile = Dataset(WindFileName)
    times_all = wrf.extract_times(WindFile, timeidx=wrf.ALL_TIMES)
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

def WRF3D(WindArray, lat, lon, hei):
    # Interpolates spacially

    # Find xy positions of the lat/lon
    ang_dist2 = (WindArray[0,0] - lat)**2 + (WindArray[1,0] - lon)**2
    min_index = np.argmin(ang_dist2); ncol = WindArray.shape[3]
    xid = min_index % ncol; yid = min_index // ncol

    lat_var = WindArray[0,0,yid-1:yid+2,xid-1:xid+2] #[3,3]
    lon_var = WindArray[1,0,yid-1:yid+2,xid-1:xid+2] #[3,3]
    hei_var = WindArray[2,:,yid-1:yid+2,xid-1:xid+2] #[z,3,3]

    # Find the variable at a certain altitude as a 2D array [3,3] #linear interpolation!
    interp_horiz = lambda entry_no: wrf.interplevel(
        field3d=WindArray[entry_no,:,yid-1:yid+2,xid-1:xid+2], 
        vert=hei_var, desiredlev=hei, missing=np.nan).data #<---RuntimeWarning originates from here

    # 2D interpolate [1,]
    latlon = np.vstack((lat_var.flatten(), lon_var.flatten())).T
    interp2pt = lambda entry_no: griddata(latlon, 
        interp_horiz(entry_no).flatten(), np.array([lat,lon]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        we = interp2pt(3)[0] # Wind east [m/s]
        wn = interp2pt(4)[0] # Wind north [m/s]
        wu = interp2pt(5)[0] # Wind up [m/s]
        tk = interp2pt(6)[0] # Temperature [K]
        pr = interp2pt(7)[0] # Pressure [Pa]
        rh = interp2pt(8)[0] # Relative humidity []
        
        
    
    wind_horizontal = LA.norm([we, wn])
    wind_direction = (-np.rad2deg(np.arctan2(wn, we)) + 270 ) % 360.

    rho_a = density_from_pressure(tk, pr, rh)
    if np.isnan(rho_a):
        return {}
    #Wind_ENU = np.vstack((we, wn, wu))
    
    
    
    return {"height": hei,
        "temperature": tk,
        "pressure": pr,
        "relative_humidity": rh,
        "wind_horizontal": wind_horizontal,
        "wind_direction": wind_direction,
        "wind_east": we,
        "wind_north": wn,
        "wind_up": wu,
        "density": rho_a}

def main(wrf_file, ref_lat, ref_lon, ref_time):
    
    ref_time = Time(ref_time)
    
    # read the data
    WindArray = WindDataExtraction(wrf_file, ref_time.jd)

    wind_dics = []
    # iterate of a range of heights
    for hei in np.arange(0., 32e3, 100.):
        wind_dics += [WRF3D(WindArray, ref_lat, ref_lon, hei)]

    # dump interpolated results to a dataframe
    df = pd.DataFrame.from_records(wind_dics).dropna()

    print(df)

    ofname = os.path.join(os.path.dirname(wrf_file),
                        f'vertical_profile_{os.path.basename(wrf_file)}_{ref_time}_{ref_lat:.5f}_{ref_lon:.5f}.csv')
    df.to_csv(ofname, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract vertical profile from WRF')
    parser.add_argument("-w", "--WRF", type=str, required=True, help="path to WRF file")
    parser.add_argument("-lat", type=str, required=True, help="latitude (decimal degrees, WGS84)")
    parser.add_argument("-lon", type=str, required=True, help="longitude (decimal degrees, WGS84)")
    parser.add_argument("-time", type=str, required=True, help="UTC time (ISO 8601 string, e.g. 2020-08-14T18:30:00)")

    args = parser.parse_args()
    
    
    ref_time = args.time
    ref_lon = float(args.lon)
    ref_lat = float(args.lat)
    
    wrf_file = args.WRF
    if not os.path.isfile(wrf_file):
        print(f'path to WRF file invalid ({wrf_file})')

    main(wrf_file, ref_lat, ref_lon, ref_time)

