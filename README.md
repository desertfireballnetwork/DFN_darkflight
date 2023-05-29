# DFN_darkflight Winchcombe

This is a snapshot of the code as it was used for the darkflight of the Winchcombe meteorite.

paper: https://arxiv.org/abs/2303.12126  https://onlinelibrary.wiley.com/doi/full/10.1111/maps.13977

call used:
```
mpirun -n 24 python DFN_DarkFlight.py -w data/profile_DN210228_02_UK_start_02-28_1200.csv -g 300 -e data/winchcombe_end_point_2022-02-11_from-Denis.cfg -mc 100
```
