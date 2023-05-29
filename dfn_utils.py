#!/usr/bin/env python
#
# Python 2 and 3 compatible as long as astropy remains 2/3 compatible
#
# Version history:
# 

from __future__ import absolute_import, division, print_function


__author__ = "Hadrien A.R Devillepoix"
__copyright__ = "Copyright 2015-2018, Desert Fireball Network"
__license__ = "MIT"
__version__ = "1.0"


import os
import sys
import re
import glob
import itertools
import logging
import subprocess
import warnings
import yaml
from datetime import datetime
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import numpy as np
from astropy.table import Table, hstack, vstack, join, Column
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.units.quantity import Quantity

# no more fancy color terminal output
fail = "[FAIL] - "
ok = "[OK] - "
warning = "[WARNING] - "
info = "[INFO] - "

## fancy color terminal output
#ffail = "\033[1;31m[FAIL]-\033[0m"
#fok = "\033[1;32m[OK]-\033[0m"
#fwarning = "\033[1;33m[WARNING]-\033[0m"
#finfo = "\033[1;36m[INFO]-\033[0m"


# default file names and extensions
rawExtensionDefault = "NEF"
fitsExtension = "fits"
tifExtension = "tiff"
jpgExtension = "jpeg"
stdcfgfilename = "dfnstation.cfg"

RAW_EXTENSIONS = {'NEF' : 'nikon',
                 'CR2' : 'canon',
                 'ARW' : 'sony'}


PROCESSING_FILTER_BAND = {'RED' : 'R',
                          'GREEN 1' : 'V',
                          'GREEN 2' : 'V',
                          'BLUE' : 'B',
                          'GREEN_INTERPOL' : 'V',
                          'RAW' : 'panchromatic',
                          'GREEN_2X2' : 'G',
                          'RGB_2X2' : 'panchromatic'}

# Can add lenses here, use lower case
# equisolid
# equidistance
# orthogonal
# stereographic
FISHEYE_LENS_PROJECTION_CATALOG = {'samyang_8mm_f3.5':'stereographic',
                                   'Oculus_1.55mm_f2': 'equisolid'}

# in millimeters
SENSOR_SIZE_CATALOG = {'Nikon D800E': (36.0, 24.0, 7424, 4924),
                       'Nikon D800': (36.0, 24.0, 7424, 4924),
                       'Nikon D810': (36.0, 24.0, 7380, 4928),
                       'Nikon D850': (36.0, 24.0, 8288, 5520),
                       'Nikon D750': (36.0, 24.0, 6032, 4032),
                       'Sony ILCE-7SM2': (36.0, 24.0, 4256, 2848),
                       'ZWO ASI1600MM Pro': (18.0, 13.5, 4656, 3520),
                       'SX Superstar allsky': (6.4, 4.75, 1392, 1040)}


class WrongTableTypeException(Exception):
    pass

class UnknownTableTypeException(Exception):
    pass

class TimingPrecisionError(Exception):
    pass
    
class DataReductionError(Exception):
    '''
    Exception class used when some pipeline task fails
    '''
    pass



    #def __init__(self, stuff, objective='undefined action'):
        #self.stuff = stuff
        #self.objective

    #def __str__(self):
        #return repr(self.stuff + ' does not have sufficient timing precision for doing ' + self.objective)
    

class AlreadyCalculatedException(Exception):
    '''03_2016-07-28_043558_K_DSC_8287-G_DN160728_01_2016-08-01_163337_dfn-user__corrected.ecsv
    Exception raised when some part of the processing pipeline has already been run on some data
    '''
    pass

    #def __init__(self, sw, data='undefined data'):
        #self.sw = sw
        #self.data = data

    #def __str__(self):
        #return repr(self.sw + ' has already been run for data: ' + self.data)
        
        
def create_logger(traj_dir, key, log_level=logging.DEBUG):

    import dfns_functions
    
    logger = logging.getLogger(key)
        
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    log_file = os.path.join(traj_dir, dfns_functions.log_name() + key + '.txt' )
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(module)s, %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    
    return log_file

def extract_event_codename_from_path(path):
    '''
    Work out the event codename from a path
    ex: 
    the path "/home/NAS_clone/events_trello/DN160131_02/blah"
    yields "DN160131_02"
    raises a ValueError if cannot find it.
    '''
    
    path_elements_list = path.split(os.sep)
    for el in path_elements_list:
        # regex to match standard dfn event naming scheme
        if re.match('DN[0-9]{6}_[0-9]{2}', el):
            return el
    else:
        raise ValueError('Event codename not found in path')
    
    

def sanitize_dictionary_for_ascii_write(dic):
    '''
    Sanitize dictionary where values are supposed to be simple types,
    but sometimes are hidden in more complex objects (quantities, times, float64...)
    '''
    for key in dic:
        if isinstance(dic[key], u.Quantity):
            dic[key] = dic[key].value
        if isinstance(dic[key], np.ndarray):
            dic[key] = dic[key][0]
        if isinstance(dic[key], np.float64):
            dic[key] = float(dic[key])
        elif isinstance(dic[key], Time):
            dic[key] = dic[key].isot

        
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        if sys.version_info < (3, 0):
            choice = raw_input().lower()
        else:
            choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def find_log_file(basedir, suffix, extension='txt', system_number=''):
    '''
    -----
    2017-06-30_DFNSMALL15_log_interval.txt
    -----
    Parameters:
        basedir: base directory where to search
        suffix: 
        extension: file extension
        system_number: 
    Returns:
        log file
    Except:
        FileNotFoundError
    '''
    #logger = logging.getLogger('trajectory')
    
    extended_suffix = system_number + suffix + '.' + extension
    
    find_comm = ['find',
                basedir,
                '-type', 'f',
                '-name', "*" + extended_suffix]
    
    list_results = [e for e in subprocess.check_output(find_comm).decode().split('\n') if extended_suffix in e and e.startswith(basedir)]
    
    if len(list_results) < 1:
        raise FileNotFoundError('Could not locate log file')
    
    return list_results[0]


def search_dfn_operation_log(log_file, key, module='', results='first'):
    '''
    Search for a key in the standard DFN logging format.
    eg. 
    -----
    2017-06-30 16:13:29,102, INFO, interval_control_lin, leostick_version, mem error fixed for now, new error notifaction system, built: 10:56:18 Apr  1 2014
    -----
    Parameters:
        log_file: path to log file
        key: keywork (eg. leostick_version on the above example)
        module (optional): module that triggered the log line (eg. interval_control_lin on the above example)
        results: type of results required ('first', 'list')
    Returns:
        value
    Except:
        KeyError
    '''
    found = False
    if results not in ['first', 'list']:
        raise KeyError('Only output types --{0}-- and --{1}-- are supported'.format('first', 'list'))
    if results == 'list':
        return_object = []
    with open(log_file, "r") as in_file:
        # Loop over each log line
        for line in in_file:
            # If log line matches our regex, print to console, and output file
            if key in line and module in line:
                parsed_value = line.split(module + ', ' + key + ', ')[1].rstrip()
                found = True
                if results == 'first':
                    return_object = parsed_value
                    break
                else:
                    return_object += [parsed_value]
    if found:
        return return_object
    else:
        raise KeyError('Could not find key {0} logged by module {1} in log file {2}'.format(key, module, log_file))

    
def get_EarthLoc(io_cfg_file):
    '''
    Create an EarthLocation object based on dfnstation.cfg file
    '''
    from astropy.coordinates import EarthLocation
    config = configparser.ConfigParser()
    config.read(io_cfg_file)
    
    # get station section
    sta = config['station']
    
    # define observer location
    return EarthLocation.from_geodetic(lat=float(sta['lat'])*u.deg,
                        lon=float(sta['lon'])*u.deg,
                        height=float(sta['altitude'])*u.meter)


def time_from_event_string(event):
    '''
    Create timestamp from a standard event codename
    Parameters:
        event: standard DFN event codename. eg. DN151024_01
    Returns:
        astropy Time object
        
    input examples:
        DN151024_01
    '''
    # for the lack of a cleanest solution:
    iso = '20' + event[2:4] + '-' + event[4:6] + '-' + event[6:8]
    
    return Time(iso)


def time_from_dfn_filename_string(ifile):
    '''
    Create timestamp from a standard dfn filename
    Parameters:
        ifile: input file (can also be a path)
    Returns:
        astropy Time object
        
    input examples:
    007_2017-08-29_121328_E_DSC_0396.thumb.jpg
    /data0/43_2014-12-30_172628_DSC_0930.NEF
    34_2016-08-30_190328-0031_video_PAL.avi
    '''
    
    fname = os.path.basename(ifile)
    
    date_block = fname.split('_')[1]
    time_block = fname.split('_')[2]
    datetime_object = datetime.strptime('{0}T{1}'.format(date_block,time_block), "%Y-%m-%dT%H%M%S")
    
    return Time(datetime_object)


def identify_calibration_files(file_list, log_file):
    '''
    Return the subset of files that are issued from calibration frames.
    Calibration frame times are logged in the interval log file by the capture control software.
    Parameters:
        file_list: list of files to test (can be basenames or full paths)
            Can take any file type as long as they start with the standard CAMCODE_YYYY-MM-DD_HHMMSS
        log_file: path to interval_log
    Return:
        subset of files
    '''
    try:
        cal_list = search_dfn_operation_log(log_file, 'next_image_calibration', module='leostick', results='list')
    except KeyError:
            return []

    cal_times = np.zeros(len(cal_list))
    for i in np.arange(len(cal_list)):
        cal_times[i] = round_to_next_30_seconds(cal_list[i]).unix

    files_that_are_from_calibration = []
    for f in file_list:
        ftime = round_to_next_30_seconds(time_from_dfn_filename_string(f)).unix
        # if the timestamp corresponds to a calibration time within 5 seconds
        if np.min(np.absolute(cal_times-ftime)) < 10.:
            files_that_are_from_calibration.append(f)

    return files_that_are_from_calibration



    
    
def time_factory(itime):
    '''
    Determine input time format and return Time object
    raises TypeError if cannot figure out format
    '''
    
    # already a Time object: return input
    if isinstance(itime, Time):
        t = itime
    # assume JD or UNIX if number of integer digits > 7
    elif isinstance(itime, float) or (isinstance(itime, str) and re.match("^\d+?\.\d+?$", itime)):
        f_itime = float(itime)
        if len(str(int(f_itime))) > 7:
            t = Time(f_itime, format='unix', scale='utc')
        else:
            t = Time(f_itime, format='jd', scale='utc')
    # assume ISOT / autodetect
    elif isinstance(itime, str):
        t = Time(itime, scale='utc')
    else:
        raise TypeError(str(itime) + 'is NOT a valid TIME input')
    
    return t



def round_to_nearest_n_seconds(itime, n):
    '''
    Returns the closest time that is the top of the minute or half minute
    Paramters:
        itime: some sort of time
        n: rounding, integer [1,60], cannot be prime with 60
    Returns:
        Corrected time stamp (astropy Time object in UTC scale)
    '''
    
    if not isinstance(n, int) or (n*int(60.0/n) != 60):
        raise TypeError("Can only round timestamp to n: integer [1,60], cannot be prime with 60")
    
    t = time_factory(itime)
    
    # 2880 = 24 * 3600 / 30
    integer_day_factor = 24 * 3600 / n
    td = TimeDelta(t.jd2 * integer_day_factor - round(t.jd2 * integer_day_factor), format='sec')
    
    return t - td * n


def round_to_nearest_30_seconds(itime):
    '''
    Returns the closest time that is the top of the minute or half minute
    # DEPRECATED
    '''
    
    #t = time_factory(itime)
    
    ## 2880 = 24 * 3600 / 30
    #td = TimeDelta(t.jd2 * 2880 - round(t.jd2 * 2880), format='sec')
    
    #return t - td * 30
    
    return round_to_nearest_n_seconds(itime, 30)



def round_to_previous_30_seconds(itime):
    '''
    Returns the previous time that is the top of the minute or half minute
    '''
    from math import floor
    
    t = time_factory(itime)
    
    # 2880 = 24 * 3600 / 30
    td = TimeDelta(t.jd2 * 2880 - floor(t.jd2 * 2880), format='sec')
    
    return t - td * 30


def round_to_next_30_seconds(itime):
    '''
    Returns the next time that is the top of the minute or half minute
    '''
    from math import ceil
    
    t = time_factory(itime)
    
    # 2880 = 24 * 3600 / 30
    td = TimeDelta(t.jd2 * 2880 - ceil(t.jd2 * 2880), format='sec')
    
    return t - td * 30


def solar_longitude(itime):
    ''' 
    The solar longitude of Earth at time, t, w.r.t. the Sun. 
    Source: Steyaert C., 1991: Calculating the solar longitude 2000.0
    '''

    t = time_factory(itime)

    
#    # Compute the Earth's position, then inverse to get the Sun's
#    Pos_ECI_SC = SkyCoord(ra=0 * u.rad, dec=0 * u.rad, distance=0 * u.m, obstime=t, frame='cirs')
#    solar_long = (Pos_ECI_SC.heliocentrictrueecliptic.lon + 180. * u.deg) % (360 * u.deg)   
        
    time_JD = t.jd
    T = (time_JD - 2451545.0) / 365250.0
    L0 = 4.8950627 + 6283.0758500 * T - 0.0000099 * T**2
    
    A = [[334166.,3480.,350.,342.,314.,268.,234.,132.,127.,120.,99.,90.,86.,78.,75.,51.,49.,
          36.,32.,28.,27.,24.,21.,21.,20.,16.,13.,13.], [20606.,430.,43.], [872.,29.], [29.]]
    B = [[4.669257,4.6261,2.744,2.829,3.628,4.418,6.135,0.742,2.037,1.110,5.233,2.045,3.508,
          1.179,2.533,4.58,4.21,2.92,5.85,1.90,0.31,0.34,4.81,1.87,2.46,0.83,3.41,1.08], 
          [2.67823,2.635,1.59], [1.073,0.44], [5.84]]
    C = [[6283.075850,12566.1517,5753.385,3.523,77713.771,7860.419,3930.210,11506.77,529.691,
          1577.344,5884.927,26.298,398.149,5223.694,5507.553,18849.23,775.52,0.07,11790.63,
          796.30,10977.08,5486.78,2544.31,5573.14,6069.78,213.30,2942.46,20.78], [6283.07585,
          12566.152,3.52], [6283.07585,12566.15], [6283.07585]]
    
    S = [0, 0, 0, 0]
    for k in range(len(A)):
        for i in range(len(A[k])):
            S[k] += A[k][i] * np.cos(B[k][i] + C[k][i] * T)
    
    # Ignoring periodical terms:
    L = L0 + (S[0] + S[1] * T + S[2] * T**2 + S[3] * T**3) * 10**-7
    
    solar_long = (L * 180/np.pi)%360 * u.deg

    return solar_long


def get_processing_software_from_meta(input_table, key):
    '''
    Parses a pipeline software identification string like: 'StraightLineLeastSquares.py 1.1'
    input_table: input astropy table
    key: key
    return: 'StraightLineLeastSquares.py', 1.1
    '''
    
    try:
        sw_meta = input_table.meta[key]
    except KeyError:
        return None, None
    
    splt = sw_meta.split(' ')
    
    if len(splt) != 2:
        raise KeyError(sw_meta + ' does not look a valid software format: <script> <version>')
    
    sw = splt[0]
    ver = splt[1]
    
    return sw, ver

def has_reliable_timing(input_table, tol=0.5):
    '''
    Determines if timing can be trusted using timing error bars
    input_table: astropy Table or row
    tol: tolerance in seconds (default = 0.1s)
    '''
    if isinstance(input_table['time_err_plus'], Quantity):
        time_err_plus = input_table['time_err_plus'].data
        time_err_minus = input_table['time_err_minus'].data
    else:
        time_err_plus = input_table['time_err_plus']
        time_err_minus = input_table['time_err_minus']
    return np.array(time_err_minus) + np.array(time_err_plus) <= tol
    # return not (max(time_err_plus) > 0.1 or
    #             max(time_err_minus) > 0.1)

def is_type_pipeline(input_table, table_type_required):
    '''
    Check table processing stage, by checking a number of columns
    input_table: astropy table
    table_type_required: processing stage
    '''
    logger = logging.getLogger('trajectory')
    
    allowed_table_types = {'point_picking' : ['x_image', 'y_image', 'de_bruijn_sequence_element_index', 'datetime', 'time_err_plus', 'time_err_minus'],
                           'astrometric' : ['altitude', 'azimuth'],
                           'triangulated' : ['longitude', 'latitude', 'height', 'X_geo', 'Y_geo', 'Z_geo', 'range'],
                           'velocitic' : ['D_DT_geo'],
                           'velocitic_modeled' : ['D_DT_EKS'],
                           'raw_photometric' : ['flux_dash_', 'brightness_dash_'],
                           'calibrated_photometric' : ['m_'],
                           'absolute_photometric' : ['M_'],
                           'key_parameters' : ['event_codename', 'duration', 'sol_long'],
                           'orbital' : ['semi_major_axis']}
    
    ttr = table_type_required.lower()
    if ttr not in allowed_table_types:
        raise UnknownTableTypeException(table_type_required + ' is not a known table type in the pipeline. Currently implemented values: ' + str(allowed_table_types))
    
    check_list = allowed_table_types[ttr]
    for col in check_list:
        if ttr == 'velocitic_modeled' and 'EKS_initial_velocity_all_cam' in input_table.meta and not np.isnan(input_table.meta['EKS_initial_velocity_all_cam']):
            # now that EKS_D_DT is no longer in tables, need to check metadata instead. 
            continue
        
        elif not any(col in s for s in input_table.colnames):
            logger.debug('current Table is NOT ' + table_type_required)
            return False

        else:
            continue
    
    # logger.debug('current Table IS ' + table_type_required)
    return True


def write_cfg_file(superdict, outputfilename):
    # write whatever needs to be written in the Observer config file
    config = configparser.RawConfigParser(allow_no_value=True)

    for cs, d in list(superdict.items()):
        config.add_section(cs)
        for k, v in list(d.items()):
            config.set(cs, k, v if v != '' else None)

    with open(outputfilename, 'w') as configfile:
        config.write(configfile)




def resolve_glob(extension, directory=".", prefix="", suffix=""):
    '''
    list files with certain pattern, bash style (directory/prefix*.extension)
    extension: simple extension (ex: "NEF"), or list of extensions
    '''
    
    # check if extension parameter is a simple extension or a list of extensions
    if isinstance(extension, str):
        reslist = sorted(glob.glob(os.path.join(directory, prefix + "*" + suffix + "*." + extension)))
    else:
        reslist = sorted(list(itertools.chain.from_iterable([glob.glob(os.path.join(directory, prefix + "*" + suffix + "*." + e)) for e in extension])))
    return reslist

def event_codename_matcher(path):
    """
    find DFN event codename in a path
    """
    
    pathElementsList = path.split(os.sep)
    for el in pathElementsList:
        # regex to match standard dfn event naming scheme
        if re.match('[A-Z]{2}[0-9]{6}_[0-9]{2}', el):
            event_codename = el
            return True, event_codename
    else:
        return False, None
                    
    ## regex to match standard dfn event naming scheme
    #if re.match('[A-Z]{2}[0-9]{6}_[A-Z]{2}[0-9]{2}', el):
        #return True, eventcodename
    ## new naming scheme, without the area number (WA, NU, SA...)
    #elif re.match('[A-Z]{2}[0-9]{6}_[0-9]{2}', el):
        #return True, eventcodename
    #return False, None

def pipeline_meta_fix(PP_file, dry_run=False):
    logger = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(module)s, %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    
    add_meta_event_name(PP_file, dry_run=dry_run)


def add_meta_event_name(PP_file, dry_run=False):
    '''
    Add event name in the metadata of a point picking file, if not already there
    Returns:
        event name found
    '''
    
    logger = logging.getLogger()
    
    if 'trajectory_' in PP_file:
        logger.debug('{0} is in a trajectory folder, skipping'.format(PP_file))
        return None
    
    try:
        PP_table = Table.read(PP_file, format='ascii.ecsv', guess=False, delimiter=',')
    except:
        logger.error('{0} is not a valid ECSV file'.format(os.path.basename(PP_file)))
        return None
    
    if not is_type_pipeline(PP_table, 'point_picking'):
        logger.debug('{0} is not a point picking table'.format(os.path.basename(PP_file)))
        return None

    goodNameFound = False
    try:
        fireballName = PP_table.meta['event_codename']
        if fireballName != 'fireball':
            goodNameFound = True
    except:
        fireballName = ''

    if goodNameFound:
        logger.debug('{0} already has event codename in metadata'.format(os.path.basename(PP_file)))
        return fireballName

    # add event name if not present already
    if fireballName in ['fireball', '', None]:
        pathElementsList = PP_file.split(os.sep)
        eventcodename = fireballName
        nameFound = False
        for el in pathElementsList:
            result_found, match = event_codename_matcher(el)
            if result_found:
                eventcodename = el
                nameFound = True
                break
            
                
        if nameFound:
            logger.info('Found an event codename for {0} : {1}. Saving result...'.format(os.path.basename(PP_file), eventcodename))
            PP_table.meta['event_codename'] = eventcodename
            if not dry_run:
                PP_table.write(PP_file, format='ascii.ecsv', delimiter=',', overwrite=True)
            fireballName = eventcodename

    return fireballName

def getDfnstationConfigFile(directory="."):
    possibleFiles = resolve_glob(extension="cfg", directory=directory, suffix="dfnstation")
    if len(possibleFiles) > 1:
        warnings.warn("Several camera config files found in the directory.")
    if os.path.join(directory, stdcfgfilename) in possibleFiles:
        return stdcfgfilename
    elif len(possibleFiles) >= 1:
        return possibleFiles[0]
    else:
        return ""


def ecsv2votable(ifile, ofile=""):
    if ofile == "":
        ofile = os.path.splitext(ifile)[0] + ".xml"
    t = Table.read(ifile, format='ascii.ecsv', delimiter=',', fill_values=[('--', 'nan')])
    t.write(ofile, format='votable', overwrite=True)
    print(ofile)


def ecsv2commHeader(ifile, ofile=""):
    if ofile == "":
        ofile = os.path.splitext(ifile)[0] + ".ascii"
    t = Table.read(ifile, format='ascii.ecsv', delimiter=',', fill_values=['--', 'nan'])
    t.write(ofile, format='ascii.commented_header', delimiter=',', fill_values=['', 'nan'], overwrite=True)
    print(ofile)

def tablefile2gnuplotready(ifile, ofile=""):
    if ofile == "":
        ofile = os.path.splitext(ifile)[0] + ".ascii"

    if ifile.endswith('.yaml'):
        raise NotImplementedError('plot yaml orbit')
        #kp = yaml.safe_load(open(ifile, 'r'))
    else:
        t = read_table(ifile)
        
    #t = Table.read(ifile, format='ascii.csv', delimiter=',')
    
    if 'event_codename' not in t.colnames:
        event_codename = ifile.split('_')[0]
        t['event_codename'] = [event_codename for i in range(len(t))]
    
    # remove no existent orbits and hyperbolics
    t_out = t[t['semi_major_axis'] > 0.0]
    t_out['event_codename','semi_major_axis','eccentricity','inclination','argument_periapsis','longitude_ascending_node','true_anomaly'].write(ofile, format='ascii.commented_header', delimiter=',', fill_values=['', 'nan'], overwrite=True)
    print(ofile)
    
def votable2commHeader(ifile, ofile=""):
    if ofile == "":
        ofile = os.path.splitext(ifile)[0] + ".ascii"
    t = Table.read(ifile, format='votable')
    
    t['event_codename','semi_major_axis','eccentricity','inclination','argument_periapsis','longitude_ascending_node','true_anomaly'].write(ofile, format='ascii.commented_header', delimiter=',', fill_values=['', 'nan'], overwrite=True)
    print(ofile)

def join_csvNecsv(ifile, ifile2):
    t = Table.read(ifile, format='ascii.csv', delimiter=',')
    if ifile2:
        t2 = Table.read(ifile2, format='ascii.ecsv', delimiter=',')
        for c in (set(t.colnames) & set(t2.colnames)):
            t2.remove_column(c)
        t = hstack([t2, t], join_type='exact')
    t.write(ifile2, format='ascii.ecsv', delimiter=',', overwrite=True)


def votable2ecsv(ifile, ofile=''):
    if ofile == "":
        ofile = os.path.splitext(ifile)[0] + ".ecsv"
    print("reading: " + ifile)
    t = Table.read(ifile, format='votable')
    t.write(ofile, format='ascii.ecsv', delimiter=',', overwrite=True)
    print(ofile)

def read_table(ifile):
    '''
    Reads in a tabular file in astropy Table objet
    Supported formats: ECSV, CSV, VOTABLE, FITS
    '''
    
    ext = os.path.splitext(ifile)[1].lower()
    
    if 'ecsv' in ext:
        table_format = 'ascii.ecsv'
        delimiter=','
        delimiter_relevant = True
    elif 'csv' in ext:
        table_format = 'ascii.csv'
        delimiter_relevant = True
        delimiter=','
    elif 'xml' in ext:
        table_format = 'votable'
        delimiter_relevant = False
    elif 'fit' in ext:
        table_format = 'fits'
        delimiter_relevant = False
    else:
        print('Unrecognised table format: ' +  ext)
        exit(1)
        
        
    if delimiter_relevant:
        t = Table.read(ifile, format=table_format, delimiter=delimiter, guess=False)
    else:
        t = Table.read(ifile, format=table_format)
        
        
    return t



def showTableInBrowser(ifile):
    '''
    Show a table in nicely formatted html
    '''    
    t = read_table(ifile)
    t.show_in_browser(jsviewer=True)
    
def tablefile_to_html(ifile):
    t = read_table(ifile)
    table_html_lines = t.pformat(html=True)
    
    header = ['<html>',
    '<head>',
    '<meta charset="utf-8"/>',
    '<meta content="text/html;charset=UTF-8" http-equiv="Content-type"/>',
    '</head>',
    '<script src="sorttable.js"></script>',
    '<body>',
    '<table class="sortable">']
    
    end = ['</body>','</html>']
    
    for s in [header, table_html_lines[1:], end]:
        [print(t) for t in s]
    
    
def print_table_info(ifile, tablefmt='rst'):
    '''
    Print statistics on a tabular file (first row, last row, median...)
    '''
    from tabulate import tabulate

    t = read_table(ifile)
    
    means = []
    maxes = []
    mins = []
    # calculate means
    for col in t.colnames:
        try:
            mins += [np.nanmin(t[col])]
            maxes += [np.nanmax(t[col])]
        except:
            mins += ['N/A']
            maxes += ['N/A']
            pass
        try:
            means += [np.mean(t[col])]
        except:
            means += ['N/A']
            pass
    
    # define a tabular structure that tabulate likes
    table = []
    for i in range(len(t.colnames)):
        if len(t) == 1:
            table += [[t.colnames[i], str(t[0][i])[:50]]]
        else:
            table += [[t.colnames[i], str(t[0][i])[:25], str(t[-1][i])[:25], str(means[i])[:25], str(mins[i])[:25], str(maxes[i])[:25]]]
    
    # call tabulate to spit it out nicely
    return {'pretty' : tabulate(table, headers=['Column', 'First row', 'Last row', 'Mean', 'Min', 'Max'], tablefmt=tablefmt),
            'table' : t}
    
    

def listKPfiles( folder, suffix='_key_parameters.csv', folderPrefix='trajectory_'):
    '''
    Key parameters files generators
    '''
    print('Listing files in: '+folder+' - with folder prefix: '+folderPrefix+' - with suffix: '+suffix)
    for root, _, files in os.walk(folder):
        #print(root)
        for filename in files:
            if filename.endswith(suffix) and any( folderPrefix in s for s in
                                                os.path.normpath(root).split(os.sep) ):
                print(root)
                print(os.path.join(root, filename))
                yield os.path.join(root, filename)



def mergeKPFiles( directory, ofile='',
                ofilePrefix='consolidated_key_parameters_file_',
                suffix='_key_parameters.csv',
                folderPrefix='trajectory_',
                save = True):
    """ load all kpfiles, merge into 1 table, save as ofile
        (csv extension)"""
    KPTableList = [Table.read(KPf, format = 'ascii.csv', delimiter = ',') 
                   for KPf in listKPfiles(directory, suffix = suffix,
                                          folderPrefix = folderPrefix)]
    #print([t['event_codename'] for t in KPTableList])
    #for t in KPTableList:
    #    print(t.colnames)
    mergedKPTable = vstack(KPTableList)

    if ofile == '':
        fname = ofilePrefix + datetime.datetime.now().strftime('%Y-%m-%d') + '.csv'
        ofile = os.path.join(directory, fname)
        ofile_cleaned = os.path.join(directory, 'cleaned_' + fname)
    if save:
        mergedKPTable.write(ofile, format='ascii.csv', delimiter=',',
                            fill_values=[('nan', '')], overwrite=True)
    print('Found ' + str(len(mergedKPTable)) + ' events in ' + folderPrefix
             + ' . stored in ' + ofile)

    cleaned_mergedKPTable = mergedKPTable[ mergedKPTable['orbit_type'] == 'Heliocentric']
    if save:
        cleaned_mergedKPTable.write(ofile_cleaned, format='ascii.csv',
                                    delimiter=',', fill_values=[('nan', '')], overwrite=True)
    print('Found ' + str(len(cleaned_mergedKPTable)) + ' events in '
            + folderPrefix + ' . stored in ' + ofile_cleaned)

    return ofile, mergedKPTable

def joinTriangulationWithIDL( directory, ofile='', date=''):
    """ .... """
    trajConsolidatedFile, trajConsolidatedTable = mergeKPFiles(directory,
                    ofilePrefix='consolidated_key_parameters_file_',
                    folderPrefix='trajectory_' + date)
    IDLTrajConsolidatedFile, IDLTrajConsolidatedTable = mergeKPFiles(directory,
                            ofilePrefix='consolidated_IDL_key_parameters_file_',
                            folderPrefix='IDL_trajectory_' + date)

    if ofile == '':
        dt = datetime.datetime.now().strftime('%Y%m%d')
        ofile = os.path.join(directory,
        'super_consolidated_key_parameters_file_' + dt + '.ecsv')

    joinedTable = join(trajConsolidatedTable, IDLTrajConsolidatedTable,
                        keys='event_codename', join_type='inner')

    joinedTable.write(ofile, format='ascii.ecsv', delimiter=',',
                        fill_values=[('nan', '')], overwrite=True)

    # print ofile

    return ofile, joinedTable


def add_JD_info(table, datetime_col_='datetime', jd_col_='JD', overwrite=True):
    
    if jd_col_ in table.colnames:
        table.remove_column(jd_col_)
    
    t = Time(table[datetime_col_].data.tolist())
    
    jd_col = Column(data=t.jd, name=jd_col_, description='Julian Date')
    
    table.add_column(jd_col)
    
    return table


def print_end_parameters(event_directory):
    """
    Return 'datetime,latitude,longitude,height' string corresponding to end point of event
    Meant to be called from sh script, check return code!
    eventpath=/home/NAS_clone/events_search/DN180212_01_Gingin
    endparams=$(python3 -c "import dfn_utils; dfn_utils.print_end_parameters(\"$eventpath\")")
    Parameters:
        event_directory: full path to event
    Prints:
        'datetime,latitude,longitude,height'
        datetime: ISO 8601 UTC
        latitude: North positive, WGS84
        longitude: East positive, WGS84
        height: in metres above WGS84 ellipsoid
    """
    
    if not os.path.isdir(event_directory):
        print('{} is not a directory'.format(event_directory))
        exit(1)
    
    auto_traj_dir_list = []
    for dir_name, _, _ in os.walk(event_directory):
        if 'trajectory_auto_' in dir_name:
            auto_traj_dir_list.append(dir_name)
    auto_traj_dir_list.sort()
    
    if len(auto_traj_dir_list) < 1:
        print('No trajectory_auto_ directory for event {}'.format(event_directory))
        exit(1)
    latest_auto_traj = os.path.join(event_directory, auto_traj_dir_list[-1])
    
    
    dir_list = os.listdir(latest_auto_traj)
    KPs = sorted([d for d in dir_list if '_key_parameters.yaml' in d])
        
    if len(KPs) < 1:
        print('Cannot find a key parameters yaml file in triangulation folder {}'.format(latest_auto_traj))
        exit(1)
        
    yaml_file = os.path.join(latest_auto_traj, KPs[-1])
    
    try:
        kp = yaml.safe_load(open(yaml_file, 'r'))
    except:
        print('Error loading key parameters file {}'.format(yaml_file))
        exit(1)
        
   
    # scroll though each camera
    try:
        dic = kp['all']
        print('{},{},{},{}'.format(dic['datetime'], dic['final_latitude'], dic['final_longitude'], dic['final_height']))
    except:
        print('Error finding info in key parameters file {}'.format(yaml_file))
        exit(1)

       
def KMLs_to_geosjon(kml_paths, ofname):
    """
    backward coimpatibility due to typo bug
    """
    KMLs_to_geojson(kml_paths, ofname)


def KMLs_to_geojson(kml_paths, ofname):
    """
    convert a list of KML files to a single geojson
    """
    import json
    from pathlib import Path
    import xml.dom.minidom as md
    import kml2geojson
    
    def read_KML(kml_path):
        # Create absolute paths
        kml_path = Path(kml_path).resolve()

        # Parse KML
        with kml_path.open(encoding='utf-8', errors='ignore') as src:
            kml_str = src.read()
        root = md.parseString(kml_str)
        return root
    
    roots = [read_KML(k) for k in kml_paths]
    
    geojson = {
      'type': 'FeatureCollection',
      'features': [],
    }
    for node in roots:
        geojson['features'].extend(kml2geojson.build_feature_collection(node, name=None)['features'])
        
    with open(ofname, 'w') as tgt:
        json.dump(geojson, tgt) 

#def getReadyForAnalysis(directory, date=''):
    #ofile, joinedTable = joinTriangulationWithIDL(directory=directory, date=date)

    #meteorites_all = extract_meteorite_droppers(joinedTable)


    #meteorites_all_file = os.path.join(directory, 'meteorites_' + os.path.basename(ofile))
    #meteorites_all.write(meteorites_all_file, format='ascii.ecsv',
                            #delimiter=',', fill_values=[('nan', '')], overwrite=True)

    #meteorites_trimmed_file = os.path.join(directory,
    #'trimmed_' + os.path.splitext(os.path.basename(meteorites_all_file))[0] + ".ascii")
    #meteorites_trimmed = meteorites_all['event_codename',
                                        #'semi_major_axis',
                                        #'eccentricity',
                                        #'inclination',
                                        #'argument_periapsis',
                                        #'longitude_ascending_node',
                                        #'true_anomaly',
                                        #'perihelion',
                                        #'aphelion']
    #meteorites_trimmed.write(meteorites_trimmed_file, format='ascii.commented_header',
                             #delimiter=',', fill_values=[('nan', '')], overwrite=True)

    #joinedTable_trimmed = joinedTable['event_codename',
                                      #'semi_major_axis',
                                      #'eccentricity',
                                      #'inclination',
                                      #'argument_periapsis',
                                      #'longitude_ascending_node',
                                      #'true_anomaly',
                                      #'perihelion',
                                      #'aphelion']
    #trimmed_file = os.path.join(directory,
                    #'trimmed_' + os.path.splitext(os.path.basename(ofile))[0] + ".ascii")
    #joinedTable_trimmed.write(trimmed_file, format='ascii.commented_header',
                              #delimiter=',', fill_values=[('nan', '')], overwrite=True)

    #joinedTable_XML_file = os.path.splitext(ofile)[0] + ".xml"
    #joinedTable.write(joinedTable_XML_file, format='votable', overwrite=True)

    # outputs:
    # trimmed_all
    # trimmed_meteorites
    # xml_all
