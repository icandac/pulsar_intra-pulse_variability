import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import simps
from matplotlib import gridspec
import statistics
from array import *
from scipy.interpolate import interp1d
from numpy import diagonal

light_time = "light_time.h5"
period = 25796

with h5py.File(light_time, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    LC_key = list(f.keys())[0]
    phase_key = list(f.keys())[1]
    time_key = list(f.keys())[2]

    # Get the data
    LC = np.array(list(f[LC_key]))
    LC = LC[:,::-1]
    phase = np.array(list(f[phase_key]))
    time = np.array(list(f[time_key]))/period

#print(LC.shape, time.shape)

# LC = LC[2580:,:]
# time=time[2580:]

####### getting diagonal LC  ##################
LC = LC[2580:,:]
init_ph = np.arange(0,1,1./52.)
LC_int = np.empty((512,512))
LC_diag = np.empty((len(time[2580:3437]),len(phase)))
for ti in range(len(time[2580:3437])):
    if ti%100 == 0:
        print('%' + str(50/67.63 + ti/len(time)) + '...')
    if ti == 3437-2600:
        print('Almost there...')
    LC_dummy = LC[ti:ti+52,:]
    for ph in range(len(phase)):
        dummy = interp1d(init_ph, LC_dummy[:,ph], fill_value="extrapolate")
        LC_int[:,ph] = dummy(phase)
    LC_diag[ti] = diagonal(LC_int)

LC = LC_diag.copy()
time=time[2580:3437]

# average

LC_ave = np.zeros((len(phase),))
for ph in range(len(phase)):
    LC_dumb = simps(LC[:,ph], x=time)/(66.62-50.0)
    LC_ave[ph] = LC_dumb

norm_const = np.max(  LC_ave)
print(norm_const)

LC_data = LC #2063
data_var = []
data_var_dummy = np.empty(512)
for dat in range(len(phase)):
    data_var_dummy = statistics.pstdev(LC_data[:,dat])
    data_var = np.append(data_var,data_var_dummy)
####################LC mesh##############################
line_a = 64.2

fig,ax1 = plt.subplots(1,figsize=(4.5,10))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1  = fig.add_subplot(111)
ax1 = plt.subplot(gs[1])#, sharex = ax0)
ax0 = plt.subplot(gs[0], sharex = ax1)

ax0.vlines(0.23, ymin=50.0, ymax=67.63, colors='red', linestyles='dashed', label='1', linewidth=1.5)
ax0.vlines(0.29, ymin=50.0, ymax=67.63, colors='red', linestyles='dashed', label='2', linewidth=1.5)
ax0.vlines(0.36, ymin=50.0, ymax=67.63, colors='red', linestyles='dashed', label='3', linewidth=1.5)
#ax0.hlines(57, xmin=0.0, xmax=1.0, colors='white', linestyles='dashed', label='a', linewidth=1.5)
#ax0.hlines(66.15, xmin=0.0, xmax=1.0, colors='white', linestyles='dashed', label='b', linewidth=1.5)
ax0.hlines(line_a, xmin=0.0, xmax=1.0, colors='white', linestyles='dashed', label='a', linewidth=1.5)
ax0.text(0.18, 53, '1', fontsize=15, color='red')
ax0.text(0.24, 53, '2', fontsize=15, color='red')
ax0.text(0.31, 53, '3', fontsize=15, color='red')
ax0.text(0.1, line_a+0.1, 'a', fontsize=15, color='white')

im = ax0.pcolormesh(phase,time,LC,cmap='jet', vmax=0.25, vmin=0)
ax0.set_ylim(ymin=50, ymax= 66.62)
cax = fig.add_axes([0.125, 0.89, 0.77, 0.01]) 
cbar = fig.colorbar(im, orientation='horizontal', cax = cax)
cbar.set_ticks([0, 0.5*norm_const, norm_const, 1.5*norm_const, 2*norm_const])
cbar.set_ticklabels([0,0.5,1,1.5,2])
cax.xaxis.set_ticks_position('top')
ax0.set_ylabel('Time [number of periods]',fontsize=15)

time_slice = int((3437-2580)*(line_a-50)/(66.62-50)) # int(3489*56/67.67)

ax1.plot(phase,  LC[time_slice,:], color='darkorange', label='line a', linewidth=2.0)
ax1.plot(phase,  LC_ave, label='average', color='black', linewidth=1.0, linestyle='-')
ax1.set_ylim(ymin = 0.0, ymax = 0.2)
ax1.set_yticks([0.25*norm_const, 0.5*norm_const, 0.75*norm_const, norm_const, 1.25*norm_const])
ax1.set_yticklabels([0.25, 0.5, 0.75, 1.0, 1.25])

ax1.fill_between(phase,   (LC_ave-data_var),   (LC_ave+data_var), alpha=0.15, facecolor='blue', label='std deviation')

ax1.set_xlabel('Phase', fontsize=15)
ax1.set_ylabel('Normalized Flux',fontsize=15)
ax1.legend(loc='upper left')#, bbox_to_anchor=(-0.01, 1.02))
ax1.set_xlim(xmin = 0.0, xmax = 1.0)
plt.subplots_adjust(hspace=0)
plt.setp(ax0.get_xticklabels(), visible=False)

plt.savefig('LCs_v3_diag-50.png', bbox_inches='tight', dpi=300)
plt.show()