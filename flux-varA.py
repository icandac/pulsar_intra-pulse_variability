import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size, std
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


check = int(0.23*512)
ave_phase = int(0.39*512)
print(check)

###### for round of observers #################
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
###############################################

LC_ave_time = np.zeros((len(phase),))
for ph in range(len(phase)):
    LC_dumb = simps(LC[:,ph], x=time)/(66.62-50.0)
    LC_ave_time[ph] = LC_dumb

norm_const = np.max(  LC_ave_time)

print("LC_time_ave = ", norm_const, ' (at phase 1)')

dummy = LC[:,check]
LC_ave = simps(dummy, time)/(66.62-50) 
print('LC_ave = ', LC_ave)

LC_data =   LC
stddev = statistics.pstdev(LC_data[:,check])
error_init  = stddev
print('Variance = ', error_init)

fig, ax = plt.subplots(1)

ax.hlines(  LC_ave,xmin=50, xmax=66.62, label='average', color='black', linewidth=1.0, linestyle='-', zorder=10) 
ax.set_ylim(ymin = 0.0, ymax=norm_const)
ax.set_yticks([LC_ave, 0.25*norm_const, 0.5*norm_const, 0.75*norm_const, norm_const])
ax.set_yticklabels(['ave', 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(xmin=50, xmax=66.62)

ax.fill_between(time, (  LC_ave-error_init), (  LC_ave+error_init), alpha=0.15, facecolor='blue', label='std deviation', zorder=5)
ax.plot(time,  LC[:,check], color='darkorange', label='line 1', linewidth=2.0, zorder=0)

ax.set_xlabel('Time [number of periods]', fontsize=18)
ax.set_ylabel('Normalized Flux',fontsize=18)

# plt.savefig('flux-var_v3.pdf', bbox_inches='tight')
plt.show()