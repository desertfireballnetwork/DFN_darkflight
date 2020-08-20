# DFN_darkflight
DFN implementation of meteor darkflight calculation

## setup
It is easiest to run in a conda environment. install anaconda or miniconda, setup conda environment:
`conda create --name darkflight_env --file df_conda_spec.txt`

additionally, you need to setup STRM.py, see comment in DFN_darkflight.py for details.

additionally you need nrlmsise atmosphere model, python interface, from:
https://github.com/DeepHorizons/Python-NRLMSISE-00.git
to provide the following files in the same folder:
```
   nrlmsise_00_data.py
   nrlmsise_00_header.py
   nrlmsise_00.py
```

## usage
edit the cfg file to match your meteorite
edit the wind profile file to match your winds

```
python DFN_darkflight.py <options>
  -h, --help            show this help message and exit
  -e EVENTFILE, --eventFile EVENTFILE
                        Event file for propagation [.ECSV, .CFG or .FITS]
  -w WINDFILE, --windFile WINDFILE
                        Wind file for the corresponding event [.CSV] (overrides cfg value)
  -v {eks,grits,raw}, --velocityModel {eks,grits,raw}
                        Specify which velocity model to use for darkflight. raw is preferred
  -m MASS, --mass MASS  Mass of the meteoroid, kg (default='fall-line')
  -d DENSITY, --density DENSITY
                        Density of the meteoroid (default=3500[kg/m3])
  -s SHAPE, --shape SHAPE
                        Specify the meteorite shape for the darkflight
                        (default=cylinder)
                        values permitted are s, c, b (for sphere,cylinder,brick) 
                        or a float for intermediate values
  -g H_GROUND, --h_ground H_GROUND
                        Height of the ground at landing site (m), float or 'a'
                        for auto, which uses SRTM data
  -k, --kml             use this option if you don't want to generate KMLs
  -J, --geojson         use this option if you want to generate geojson (must
                        have KMLs as well
  -K TRAJECTORYKEYWORD, --trajectorykeyword TRAJECTORYKEYWORD
                        Add a personalised keyword to output trajectory folder name.
  -mc MONTECARLO, --MonteCarlo MONTECARLO
                        Number of Monte Carlo simulations for the darkflight (overrides cfg value)
  -me MASS_ERR, --mass_err MASS_ERR
                        mass error range as 1-x,1+x multiplier to -m for MC
                        (default=0.1)
  -se SHAPE_ERR, --shape_err SHAPE_ERR
                        shape error as +/- for MC (default=0.15)
  -we WIND_ERR, --wind_err WIND_ERR
                        wind magnitude error in each layer as +/- for MC
                        (default=2.0 m/s)
```

## notes

for monte carlo:
`-s 1.4 -se 0.15` will simulate drags of ~1.2 to ~1.6, matching the drag range of sphere to rounded brick

If your system is set up appropriately, you can do:
`mpirun -n N python DFN_darkflight.py <options> -mc 1000`
for example, where the 1000 mc runs will be split accross N cores on a multicore processor. It's faster!

`-g a` will use Shuttle Radar Topography (STRM) to get terrain heights, and automatically set ground height.
STRM covers +/-52 deg latitude. The ASTER dataset is file compatible, and has higher latitude coverage.
You will need to download the appropriate files, and proably tweak the code, but file formats are the same.

