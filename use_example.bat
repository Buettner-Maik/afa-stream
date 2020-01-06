@echo off
rem -d to specify data sets
rem -p to do preprocessing [shuffle] [batch size] [ild size]
rem -t to specify tasks [lower, upper, [RA, SWAED, MWAED, SWIG, SWSU]+[SBM,IPF,NBM]]
rem -i to specify iterations [rangeStart..rangeEnd] i.e. '0..1' for iteration 0
rem -m to specify missingness' [missingness]
rem -b to specify budgets [budget]
rem -w to specify window [size in batches]
rem -l to specify log [log path or log name]
start python additions\data_prepare.py -d abalone adult magic nursery occupancy pendigits sea -p True 50 50 -t lower upper RA+SBM SWAED+IPF SWAED+SBM SWIG+IPF SWIG+SBM SWSU+IPF SWSU+SBM MWAED+IPF MWAED+SBM -i 0..3 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -b 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 -w 10 -l tasks1.log
start python additions\data_prepare.py -d abalone adult magic nursery occupancy pendigits sea -p True 50 50 -t lower upper RA+SBM SWAED+IPF SWAED+SBM SWIG+IPF SWIG+SBM SWSU+IPF SWSU+SBM MWAED+IPF MWAED+SBM -i 3..6 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -b 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 -w 10 -l tasks2.log
start python additions\data_prepare.py -d abalone adult magic nursery occupancy pendigits sea -p True 50 50 -t lower upper RA+SBM SWAED+IPF SWAED+SBM SWIG+IPF SWIG+SBM SWSU+IPF SWSU+SBM MWAED+IPF MWAED+SBM -i 6..8 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -b 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 -w 10 -l tasks3.log
start python additions\data_prepare.py -d abalone adult magic nursery occupancy pendigits sea -p True 50 50 -t lower upper RA+SBM SWAED+IPF SWAED+SBM SWIG+IPF SWIG+SBM SWSU+IPF SWSU+SBM MWAED+IPF MWAED+SBM -i 8..10 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -b 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 -w 10 -l tasks4.log
cmd /k