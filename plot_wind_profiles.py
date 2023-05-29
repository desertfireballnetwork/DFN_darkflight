__author__ = "Hadrien A R Devillepoix"
__license__ = "MIT"
__version__ = "1.0"

import os
import glob

from astropy.time import Time
from astropy.table import Table
import numpy as np

import itertools
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns


def base_profile_name(i):
    
    o = "_".join(os.path.basename(i).split('_')[3:6])
    return o.split(':')[0]

def plot(ifiles):
    
    
    
    tables = [Table.read(t) for t in ifiles]
    for i in range(len(tables)):
        tables[i].meta['profile_ID'] = base_profile_name(ifiles[i])
    

    #fig, ax1 = plt.subplots()
    fig = plt.figure(figsize=(14.14,10))

    ax1 = plt.subplot(121)

    ax2 = plt.subplot(122)

    #ax2 = ax1.twinx()

    palette = itertools.cycle(sns.color_palette())

    fsize = 12
    matplotlib.rcParams.update({'font.size': fsize})

    for table in tables:
        color = next(palette)
        code = ""
        ax1.plot(table['wind_direction'],
                    table['height'] / 1000.,
                    '-x',
                    color=color)
        
        ax2.plot(table['wind_horizontal'],
                    table['height'] / 1000.,
                    '-x',
                    color=color,
                    label=table.meta['profile_ID'])
    ax1.set_xlim(0,360)

    ax1.set_ylabel("Height (km)", fontsize=fsize)
    ax1.set_xlabel("Wind direction (degrees E of N)", fontsize=fsize)
    
    ax2.set_ylabel("Height (km)", fontsize=fsize)
    ax2.set_xlabel("Wind speed (m/s)", fontsize=fsize)

    #for axtixlist in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks(), ax2.xaxis.get_major_ticks(), ax2.yaxis.get_major_ticks()]:
        #for tick in axtixlist:
            #tick.label.set_fontsize(fsize)

    #ax1.invert_xaxis()
    #ax2.invert_xaxis()


    ax1.grid()
    ax2.grid()
    
    ax2.legend()
#    lgd = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
    # loc='upper center',loc='upper center'fancybox=True,

    dirname = os.path.dirname(ifiles[0])
    event_codename = 'unknown_event'
    for e in dirname.split('/'):
        if e.startswith('DN'):
            event_codename = e
            break
    
    ofile = os.path.join(dirname, event_codename + '_' + str(len(ifiles)) + '_wind_profiles_comparison.pdf')
    print(f'saving to {ofile}')
    plt.savefig(ofile, bbox_inches='tight')

    #plt.show()

    


def main(idir):
    
    pattern = "vertical_profile_wrfout_*.csv"
    ifiles = glob.glob(os.path.join(idir, pattern))
    
    print('input files:')
    for i in ifiles:
        print(os.path.basename(i))
        
    if len(ifiles) < 1:
        print(f'No files matching pattern {os.path.join(idir, pattern)}')
        
    plot(ifiles)
    
    # vertical_profile_wrfout_d03_2022-10-23_19-35-02_2022-10-23T19:34:03.000_51.78012_-1.83055.csv

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot wind profiles for comparison')
    parser.add_argument("-d", "--directory", type=str, required=True, help="path to folder where profiles are stored")

    args = parser.parse_args()
    
    
    idir = args.directory

    main(idir)

