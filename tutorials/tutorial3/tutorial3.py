#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Current, parent and file paths
CWD = os.getcwd()
CFD = os.path.abspath('')

# project libraries
sys.path.append(os.path.join(CFD,"../../"))

# Import library specific modules
from pyspod.spod.standard  import Standard  as spod_standard # for use of multitaper method
from pyspod.spod.multitaper  import Standard  as spod_multitaper # for use of multitaper method
import pyspod.spod.utils     as utils_spod
import pyspod.utils.weights  as utils_weights
import pyspod.utils.errors   as utils_errors
import pyspod.utils.io       as utils_io
import pyspod.utils.postproc as post



## -------------------------------------------------------------------
## initialize MPI
## -------------------------------------------------------------------
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
except:
    comm = None
    rank = 0
## -------------------------------------------------------------------



## -------------------------------------------------------------------
## read data and params
## -------------------------------------------------------------------
## data
data_file = os.path.join(CFD, '../../tests/data/', 'era_interim_data.nc')
ds = utils_io.read_data(data_file=data_file)
print(ds)
## we extract time, longitude and latitude
t = np.array(ds['time'])
x1 = np.array(ds['longitude']) - 180
x2 = np.array(ds['latitude'])
data = ds['tp']
nt = len(t)
print('shape of t (time): ', t.shape)
print('shape of x1 (longitude): ', x1.shape)
print('shape of x2 (latitude) : ', x2.shape)
## params
config_file = os.path.join(CFD, '../../tests/data', 'input_tutorial2.yaml')
params = utils_io.read_config(config_file)
## set weights
weights = utils_weights.geo_trapz_2D(
    x1_dim=x2.shape[0], x2_dim=x1.shape[0],
    n_vars=params['n_variables'])
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## compute spod using Welch's method
## -------------------------------------------------------------------
standard  = spod_standard(params=params, weights=weights, comm = comm)
spod_stad = standard.fit(data_list=data)
standard_results_dir = spod_stad.savedir_sim
spod_stad.plot_eigs_vs_period()
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## compute spod using Multitaper-Welch method
## -------------------------------------------------------------------
params['half_bandwidth'] = 2   # by default it will set n_taper = 10
# params['n_tapers'] = 3  # number of tapers can also be customized
multitaper = spod_multitaper(params=params, weights=weights, comm = comm)
spod_mltp = multitaper.fit(data_list=data)
multitaper_results_dir = spod_mltp.savedir_sim
spod_mltp.plot_eigs_vs_period()

## -------------------------------------------------------------------
## SPOD mode comparison
## -------------------------------------------------------------------
T = 1008 # hours, typical for MJO
f_stad, f_stad_idx = spod_stad.find_nearest_freq(freq_req=1/T, freq=spod_stad.freq)
spod_stad.plot_2d_modes_at_frequency(freq_req=f_stad, freq=spod_stad.freq,
    modes_idx=[0,1], x1=x1, x2=x2, coastlines='centred',
    equal_axes=True)

f_mltp, f_mltp_idx = spod_mltp.find_nearest_freq(freq_req=1/T, freq=spod_mltp.freq)
spod_mltp.plot_2d_modes_at_frequency(freq_req=f_mltp, freq=spod_mltp.freq,
    modes_idx=[0,1], x1=x1, x2=x2, coastlines='centred',
    equal_axes=True)