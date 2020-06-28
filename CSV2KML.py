#!/usr/bin/env python2.7
"""
=============== Converts a CSV to KML ===============
Created on Mon Jul 06 12:18:06 2015
@author: Trent Jansen-Sturgeon

"""
import os
import logging

import numpy as np
from astropy.table import Table

NAMESPACE = "http://earth.google.com/kml/2.2"


def fetchTriangdata(ifile):
    if ifile.split('.')[-1] == 'csv':
        triangulation_table = Table.read(ifile, format='ascii.csv', guess=False, delimiter=',')
    elif ifile.split('.')[-1] == 'ecsv':
        triangulation_table = Table.read(ifile, format='ascii.ecsv', guess=False, delimiter=',')
    elif ifile.split('.')[-1] == 'fits':
        from astropy.io import fits
        triangulation_table = Table(fits.open(ifile, mode='append')[-1].data)
    else:
        print('Unknown file format: ', ifile.split('.')[-1])

    try:
        cam_name = triangulation_table.meta['telescope'] \
            + " " + triangulation_table.meta['location']
    
        if 'CUT_TOP' in ifile:
            cam_name += ' CUT TOP'
        elif 'CUT_BOTTOM' in ifile:
            cam_name += ' CUT BOTTOM'
            
    except KeyError:
        cam_name = ''

    return triangulation_table, cam_name


def Path(FileName):
    
    [Data, cam_name] = fetchTriangdata(FileName)
    outputname = os.path.join(os.path.dirname(FileName), '.'.join(os.path.basename(FileName).split('.')[:-1]) + '_path.kml')

    # Extract some data for kml
    lat = Data['latitude'] #deg
    lon = Data['longitude'] #deg
    hei = Data['height'] #m

    return write_path_kml(lat, lon, hei, cam_name, outputname)

def write_path_kml(lat, lon, hei, cam_name, outputname):

    # Open the file to be written.
    f = open(outputname, 'w')

    # Writing the kml file.
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n')
    f.write('<Document>\n')
    f.write('	<name>' + outputname + '</name>\n')

    f.write('	<Style id="s_ylw-pushpin_hl">\n')
    f.write('		<IconStyle>\n')
    f.write('			<color>ffff0000</color>\n')
    f.write('			<scale>0.472727</scale>\n')
    f.write('			<Icon>\n')
    f.write('				<href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png</href>\n')
    f.write('			</Icon>\n')
    f.write('		</IconStyle>\n')
    f.write('		<ListStyle>\n')
    f.write('		</ListStyle>\n')
    f.write('		<LineStyle>\n')
    f.write('			<color>ff000000</color>\n')
    f.write('		</LineStyle>\n')
    f.write('		<PolyStyle>\n')
    f.write('			<color>66ffffff</color>\n')
    f.write('		</PolyStyle>\n')
    f.write('	</Style>\n')
    f.write('	<StyleMap id="m_ylw-pushpin">\n')
    f.write('		<Pair>\n')
    f.write('			<key>normal</key>\n')
    f.write('			<styleUrl>#s_ylw-pushpin</styleUrl>\n')
    f.write('		</Pair>\n')
    f.write('		<Pair>\n')
    f.write('			<key>highlight</key>\n')
    f.write('			<styleUrl>#s_ylw-pushpin_hl</styleUrl>\n')
    f.write('		</Pair>\n')
    f.write('	</StyleMap>\n')
    f.write('	<Style id="s_ylw-pushpin">\n')
    f.write('		<IconStyle>\n')
    f.write('			<color>ffff0000</color>\n')
    f.write('			<scale>0.4</scale>\n')
    f.write('			<Icon>\n')
    f.write('				<href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>\n')
    f.write('			</Icon>\n')
    f.write('		</IconStyle>\n')
    f.write('		<ListStyle>\n')
    f.write('		</ListStyle>\n')
    f.write('		<LineStyle>\n')
    f.write('			<color>ff000000</color>\n')
    f.write('		</LineStyle>\n')
    f.write('		<PolyStyle>\n')
    f.write('			<color>66ffffff</color>\n')
    f.write('		</PolyStyle>\n')
    f.write('	</Style>\n')

    f.write('	<Placemark>\n')
    #f.write('		<name>' + FileName.split('.')[0] + '</name>\n')
    f.write('		<open>1</open>\n')
    f.write('		<styleUrl>#m_ylw-pushpin</styleUrl>\n')
    f.write('		<LineString>\n')
    f.write('			<extrude>1</extrude>\n')
    f.write('			<tessellate>1</tessellate>\n')
    f.write('			<altitudeMode>absolute</altitudeMode>\n')
    f.write('			<coordinates>\n')
    # print Data
    for i in range(len(hei)):
        if not np.isnan(hei[i]):
            f.write('					' + str(lon[i]) + ',' +
                    str(lat[i]) + ',' + str(hei[i]) + '\n')
    f.write('			</coordinates>\n')
    f.write('		</LineString>\n')
    f.write('	</Placemark>\n')

    f.write('</Document>\n')
    f.write('</kml>\n')
    f.close()
    
    return {'fname' : outputname,
            'kml_type' : 'path',
            'camera' : cam_name}
            


def Points(FileName, label='NoLabel', colour='ff00ffff'):
    
    [Data, cam_name] = fetchTriangdata(FileName)  
    outputname = os.path.join(os.path.dirname(FileName), '.'.join(os.path.basename(FileName).split('.')[:-1]) + '_points.kml')

    # Extract some data for kml
    lat = Data['latitude'] #deg
    lon = Data['longitude'] #deg
    hei = Data['height'] #m
    datetime = Data['datetime']

    return write_points_kml(lat, lon, hei, datetime, cam_name, outputname, label, colour)

def write_points_kml(lat, lon, hei, datetime, cam_name, outputname, label='NoLabel', colour='ff00ffff'):

    # Open the file to be written.
    f = open(outputname, 'w')

    # Writing the kml file.
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2"' +
            ' xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n')
    f.write('<Document>\n')
    f.write('	<name>' + outputname + '</name>\n')
    f.write('	<StyleMap id="m_ylw-pushpin">\n')
    f.write('		<Pair>\n')
    f.write('			<key>normal</key>\n')
    f.write('			<styleUrl>#s_ylw-pushpin</styleUrl>\n')
    f.write('		</Pair>\n')
    f.write('		<Pair>\n')
    f.write('			<key>highlight</key>\n')
    f.write('			<styleUrl>#s_ylw-pushpin_hl</styleUrl>\n')
    f.write('		</Pair>\n')
    f.write('	</StyleMap>\n')
    f.write('	<Style id="s_ylw-pushpin">\n')
    f.write('		<IconStyle>\n')
    f.write('			<color> '+ str(colour) + '</color>\n')
    f.write('			<scale>1</scale>\n')
    f.write('			<Icon>\n')
    f.write('				<href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>\n')
    f.write('			</Icon>\n')
    f.write('		</IconStyle>\n')
    f.write('		<ListStyle>\n')
    f.write('		</ListStyle>\n')
    f.write('	</Style>\n')
    f.write('	<Style id="s_ylw-pushpin_hl">\n')
    f.write('		<IconStyle>\n')
    f.write('			<color>'+ str(colour) + '</color>\n')
    f.write('			<scale>0.472727</scale>\n')
    f.write('			<Icon>\n')
    f.write('				<href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png</href>\n')
    f.write('			</Icon>\n')
    f.write('		</IconStyle>\n')
    f.write('		<ListStyle>\n')
    f.write('		</ListStyle>\n')
    f.write('	</Style>\n')

    i = 0
    for i in range(len(hei)):
        if not np.isnan(hei[i]):
            f.write('		<Placemark>\n')
            if not isinstance(label, str):
                f.write('			<name>' + str(label[i]) + 'kg' + '</name>\n')
            f.write('			<description>' + str(datetime[i]) + '</description>\n')
            f.write('			<styleUrl>#m_ylw-pushpin</styleUrl>\n')
            f.write('			<Point>\n')
            f.write('				<altitudeMode>absolute</altitudeMode>\n')
            f.write('				<coordinates>' + str(lon[i]) + "," +
                    str(lat[i]) + "," + str(hei[i]) + '</coordinates>\n')
            f.write('			</Point>\n')
            f.write('		</Placemark>\n')
            i += 1
        
    f.write('</Document>\n')
    f.write('</kml>\n')
    f.close()
    
    return {'fname' : outputname,
            'kml_type' : 'points',
            'camera' : cam_name}


def Projection(FileName, colour='33ff0000'):

    [Data, cam_name] = fetchTriangdata(FileName)
    Long0 = Data.meta['obs_longitude']
    Lat0 = Data.meta['obs_latitude']
    H0 = Data.meta['obs_elevation']

    # Open the file to be written.
    outputname = os.path.join(os.path.dirname(FileName), os.path.basename(FileName).split('.')[0] + '_camera.kml')
    f = open(outputname, 'w')

    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://earth.google.com/kml/2.2">\n')
    f.write('<Document>\n')
    #f.write('<name>' + FileName.split('.')[0] + '_camera' + '</name>\n')
    f.write('<open>1</open>\n')
    f.write('<Placemark>\n')
    f.write('	<Style id="camera">\n')
    f.write('		<LineStyle>\n')
    f.write('			<width>1</width>\n')
    f.write('		</LineStyle>\n')
    f.write('		<PolyStyle>\n')
    f.write('			<color>' + str(colour) + '</color>\n')
    f.write('		</PolyStyle>\n')
    f.write('	</Style>\n')
    f.write('	<styleUrl>#camera</styleUrl>\n')
    f.write('<name>' + str(cam_name) + '</name>\n')
    f.write('<Polygon>\n')
    f.write('	<extrude>0</extrude>\n')
    f.write('	<altitudeMode>absolute</altitudeMode>\n')
    f.write('	<outerBoundaryIs>\n')
    f.write('		<LinearRing>\n')
    f.write('		<coordinates>\n')
    f.write('			' + str(Long0) + ',' + str(Lat0) + ',' + str(H0) + '\n')
    # for row in [Data[0],Data[-1]]:
    for row in Data:
        f.write('				' + str(row['longitude']) + "," +
                str(row['latitude']) + "," + str(row['height']) + '\n')
    f.write('			' + str(Long0) + ',' + str(Lat0) + ',' + str(H0) + '\n')
    f.write('		</coordinates>\n')
    f.write('		</LinearRing>\n')
    f.write('	</outerBoundaryIs>\n')
    f.write('</Polygon>\n')
    f.write('</Placemark>\n')
    f.write('</Document>\n')
    f.write('</kml>\n')

    f.close()
    
    return {'fname' : outputname,
            'kml_type' : 'projection',
            'camera' : cam_name}


from trajectory_utilities import ENU2ECEF, LLH2ECEF, ECEF2LLH
def Rays(FileName,  height_cutoff=100e3, colour='33ff0000'):
    # Create KML with line of sights
    
    [Data, cam_name] = fetchTriangdata(FileName)
    obs_lon = np.deg2rad(Data.meta['obs_longitude'])
    obs_lat = np.deg2rad(Data.meta['obs_latitude'])
    obs_hei = Data.meta['obs_elevation']
    
    ''' Converts alt/az to lat/lon/hei '''
    # Extract some raw data
    alt = np.deg2rad(Data['altitude'].data)
    azi = np.deg2rad(Data['azimuth'].data)

    try:
        dist = Data['range'].data
    except KeyError:
        dist = height_cutoff / np.sin(alt)

    outputname = os.path.join(os.path.dirname(FileName), os.path.basename(FileName).split('.')[0] + '_rays.kml')

    return write_rays_kml(obs_lat, obs_lon, obs_hei, alt, azi, dist, cam_name, outputname, colour)

def write_rays_kml(obs_lat, obs_lon, obs_hei, alt, azi, dist, cam_name, outputname, colour='33ff0000'):
    
    # Convert from spherical to cartesian 
    UV_ENU = np.vstack((np.cos(alt) * np.sin(azi),
                        np.cos(alt) * np.cos(azi),
                        np.sin(alt)))
    
    UV_ECEF = ENU2ECEF(obs_lon, obs_lat).dot(UV_ENU)
    Obs_ECEF = LLH2ECEF(np.vstack((obs_lat, obs_lon, obs_hei)))

    Proj_ECEF = Obs_ECEF + dist * UV_ECEF
    Proj_LLH  = ECEF2LLH(Proj_ECEF)

    [Proj_lat, Proj_lon, Proj_hei] = [Proj_LLH[0], Proj_LLH[1], Proj_LLH[2]]
    
    # Open the file to be written.
    f = open(outputname, 'w')

    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://earth.google.com/kml/2.1">\n')
    f.write('<Document>\n')
    f.write('<open>1</open>\n')
    f.write('<Placemark>\n')
    f.write('    <Style id="camera">\n')
    f.write('        <LineStyle>\n')
    f.write('            <width>1</width>\n')
    f.write('        </LineStyle>\n')
    f.write('        <PolyStyle>\n')
    f.write('            <color>' + str(colour) + '</color>\n')
    f.write('        </PolyStyle>\n')
    f.write('    </Style>\n')
    f.write('    <styleUrl>#camera</styleUrl>\n')
    f.write('<name>' + str(cam_name) + '</name>\n')
    f.write('<Polygon>\n')
    f.write('    <extrude>0</extrude>\n')
    f.write('    <altitudeMode>absolute</altitudeMode>\n')
    f.write('    <outerBoundaryIs>\n')
    f.write('        <LinearRing>\n')
    f.write('        <coordinates>\n')
    f.write('            ' + str(np.rad2deg(obs_lon)) + ',' 
        + str(np.rad2deg(obs_lat)) + ',' + str(obs_hei) + '\n')
    for i in range(len(Proj_lat)):
        f.write('                ' + str(np.rad2deg(Proj_lon[i])) + "," +
                str(np.rad2deg(Proj_lat[i])) + "," + str(Proj_hei[i]) + '\n')
        f.write('                ' + str(np.rad2deg(obs_lon)) + ',' + 
                str(np.rad2deg(obs_lat)) + ',' + str(obs_hei) + '\n')
    f.write('        </coordinates>\n')
    f.write('        </LinearRing>\n')
    f.write('    </outerBoundaryIs>\n')
    f.write('</Polygon>\n')
    f.write('</Placemark>\n')
    f.write('</Document>\n')
    f.write('</kml>\n')

    f.close()

    return {'fname' : outputname,
            'kml_type' : 'rays', 
            'camera' : cam_name}


def merge_trajectory_KMLs(kml_files_collections, out_kmz):
    import subprocess
    
    logger = logging.getLogger('trajectory')
    
    uniq_cameras = set([e['camera'] for e in kml_files_collections])
    #print(uniq_cameras)
    kmls = [e['fname'] for e in kml_files_collections]
    
    # create doc.kml
    doc_kml_file = os.path.join(os.path.dirname(out_kmz), 'doc.kml')
    with open(doc_kml_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f.write('<Document>\n')
        for cam in list(uniq_cameras):
            f.write('<Folder>\n')
            f.write('<name>' + cam + '</name>\n')
            for elem in kml_files_collections:
                if elem['camera'] == cam:
                    f.write('<NetworkLink>\n')
                    if 'CUT' in cam:
                        f.write('<visibility>0</visibility>\n')
                    f.write('<name>' + elem['kml_type'] + '</name>\n')
                    f.write('<Link>\n')
                    f.write('<href>' + os.path.basename(elem['fname']) + '</href>\n')
                    f.write('</Link>\n')
                    f.write('</NetworkLink>\n')
            f.write('</Folder>\n')
                    
        f.write('</Document>\n')
        f.write('</kml>\n')
    
    # clean destination
    try:
        os.remove(out_kmz)
    except:
        pass
    
    logging.info('Zipping {} KMLs into {}'.format(len(kmls), out_kmz))
    subprocess.check_output(["zip", "--junk-paths", "-r", out_kmz, doc_kml_file] + kmls)
    
    try:
        os.remove(doc_kml_file)
    except:
        logging.error('Problem removing {}'.format(doc_kml_file))



    

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    Projection(filename)
    Path(filename)
    Points(filename)
