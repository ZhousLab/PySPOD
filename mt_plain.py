# -*- coding: utf-8 -*-
import pyspod.spod as spod
from pyspod.spod.multitaper import Standard  as mult_standard
import numpy as np
import matplotlib.pyplot as plt
import os



def jet_plot(x,y,z):
    plt.figure(figsize = (10,3), dpi = 250)
    # plt.contourf(x,r,p[:,:,100]-p_mean, levels = 100)
    plt.contourf(x,y,z, levels = 100)
    plt.xlim(0,20)
    plt.ylim(0,3)
    cb = plt.colorbar()
    cb.set_label(r'$\bar{p}$',size = '16')



# 1. load data
path = 'data//'
dt = np.load(path+'dt.npy')
p = np.load(path+'p.npy')
x = np.load(path+'x.npy')
r = np.load(path+'r.npy')
p_mean = np.load(path+'p_mean.npy')
# p1 = p.transpose(2,0,1)
# np.save(path + 'p.npy',p1)
# p = p-p_mean

jet_plot(x, r, p[1,:,:])
nFFT = 2048
f_plot = 0.24
# 2. SPOD calcualtion
# 2.1 Params and weights
# define required and optional parameters
n_var = 1
params = dict()

params['time_step'   ] = 0.2              # 1 day, data time-sampling
params['n_snapshots' ] = p.shape[0]          # number of time snapshots (we consider all data)
params['n_space_dims'] = 2                  # number of spatial dimensions (longitude and latitude)
params['n_variables' ] = n_var              # number of variables
params['n_dft'       ] = 2048         


# -- optional parameters
params['overlap'          ] = 50            # dimension block overlap region
params['mean_type'        ] = 'longtime'   # type of mean to subtract to the data
params['normalize_weights'] = False      # normalization of weights by data variance
params['normalize_data'   ] = False        # normalize data by data variance
params['n_modes_save'     ] = 5             # modes to be saved
params['conf_level'       ] = 0.95          # calculate confidence level
params['reuse_blocks'     ] = False         # whether to reuse blocks if present
params['savefft'          ] = False         # save FFT blocks to reuse them in the future (saves time)
params['savedir'          ] = os.path.join('results\\modes') # folder where to save results

# -- taper !!!
# params['n_tapers'] = 10  # number of tapers
params['half_bandwidth'] = 5.5   # bandwidth
# standard  = mult_standard(params=params)
standard  = mult_standard(params=params)
spod = standard.fit(data_list=p)
spod.plot_eigs_vs_frequency()
#%%
# Q_blk
# q0 = Q_blk_hat_taper[:,1022:1028]
# Q_blk_hat 
# q0 = Q_blk_hat[:,0:3,0]
# q1 = Q_blk_hat[:,0:3,1]
# q2 = Q_blk_hat[:,0:3,2]
# q3 = Q_blk_hat[:,0:3,3]
# Q_hat 
# q0 = Q_hat[:,0:3,0]
# q1 = Q_hat[:,0:3,1]
# q2 = Q_hat[:,0:3,2]
# q3 = Q_hat[:,0:3,3]
#%%
l = spod.eigs
f = spod.freq
rank = 3
plt.figure(figsize = (4,3), dpi = 250)
for i in range(l.shape[1]):
    if i < rank:
        plt.loglog(f,l[:,i], linewidth = 0.8)
    else:
        plt.loglog(f,l[:,i], color = 'lightgrey', linewidth = 0.5)
plt.ylim(1e-12,1e-2)
plt.axvline(0.24, color = 'grey', linestyle = '--', linewidth = 0.8)
plt.xlabel('Frequency')
plt.ylabel('SPOD mode energy')
#%%
f1, f1_idx = spod.find_nearest_freq(freq_req=0.24, freq=spod.freq)
dirs = standard.modes_dir
modepath = os.path.join(dirs, 'freq_idx_00000098.npy')
modes = np.load(modepath)

# def vrange(data):
#     m =abs(data).max
#     return m
fig, axs = plt.subplots(3, 1, figsize = (5,5), dpi = 250)
# fig.figure(figsize = (4,4), dpi = 250)
axs[0].contourf(x,r,modes[:,:,0,0].real,11, vmin = -1*abs(modes[:,:,0,0].real).max(), vmax = abs(modes[:,:,0,0].real).max())
axs[0].axis('scaled')
axs[0].set_xlim(0,10)
axs[0].set_ylim(0,2)
axs[0].set_xticks(np.arange(0, 11, 1))

axs[1].contourf(x,r,modes[:,:,0,1].real,11, vmin = -1*abs(modes[:,:,0,1].real).max(), vmax = abs(modes[:,:,0,1].real).max())
axs[1].axis('scaled')
axs[1].set_xlim(0,10)
axs[1].set_ylim(0,2)
axs[1].set_xticks(np.arange(0, 11, 1))

axs[2].contourf(x,r,modes[:,:,0,2].real, 11, vmin = -1*abs(modes[:,:,0,2].real).max(), vmax = abs(modes[:,:,0,2].real).max())
axs[2].axis('scaled')
axs[2].set_xlim(0,10)
axs[2].set_ylim(0,2)
axs[2].set_xticks(np.arange(0, 11, 1))
#%%
import matplotlib.pyplot as plt

plt.matshow(s.real)
