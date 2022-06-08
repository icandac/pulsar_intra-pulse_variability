import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks, peak_widths, find_peaks_cwt, argrelextrema, peak_prominences
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
from scipy.interpolate import interp1d
from numpy import diagonal
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors
from scipy.integrate import simps
from scipy.stats import norm
from statistics import stdev, pstdev, variance, pvariance

dir="../data/"

#===========================================================================
# Parameters of the simulation
params1=np.loadtxt(dir+"phys_params.dat",skiprows=1)
params2=np.loadtxt(dir+"input_params.dat",skiprows=1)
rmax=params2[11]
rpml=params2[12]
rlc=params2[13]
print("Loading data")
#===========================================================================
light_time = dir+"light_time.h5"
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

# ######## for round of observers ################
# LC=LC[2580:,:]
# time=time[2580:]
# ######## getting diagonal LC  ##################
LC=LC[2580:,:]
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
###################################################

FLEs = 90
FLEe = 205
SLEs = 340
SLEe = 463
maxi_listMIN = [[]]
maxi_listFP = [[]]
maxi_listPROM = [[]]
maxi_list3 = [[]]
mini_list = [[]]
Width_t = []
Pprom_t = []
N_pl = np.empty((2,(3437-2580)), dtype=int)

# print(LC[:,0].shape, time.shape)
LC_ave = np.empty(len(phase))
for ph in range(len(phase)):
    LC_ave[ph] = simps(LC[:,ph], time)/(66.62-50.0)#(67.63-50.0)

norm_const = np.max(LC_ave)
LC_ave = np.concatenate((np.zeros(FLEs),LC_ave[FLEs:FLEe],np.zeros(SLEs-FLEe),LC_ave[SLEs:SLEe],np.zeros(512-SLEe)))

LC_time = np.empty((len(LC[:,0]),len(phase)))
LC_check = np.empty((len(LC[:,0]),len(phase)))
LC_time_all = np.empty((len(LC[:,0]),len(phase)))
LC_check_all = np.empty((len(LC[:,0]),len(phase)))

width_lim = 1.e-3*512
prom_lim = 2.e-4
# width_lim = 2.5e-2*512
# prom_lim = 8.e-2

print("LC is set.")

for step in range(1,(3437-2580)):
    # print(step)
    LC_time_all[step] = 4*(LC[step,:])
    LC_check_all[step] = np.concatenate((np.zeros(FLEs),LC_time[step][FLEs:FLEe],np.zeros(SLEs-FLEe),LC_time[step][SLEs:SLEe],np.zeros(512-SLEe)))
    LC_time[step] = 4*(LC[step,:] - LC_ave)# - data_var)
    LC_check[step] = np.concatenate((np.zeros(FLEs),LC_time[step][FLEs:FLEe],np.zeros(SLEs-FLEe),LC_time[step][SLEs:SLEe],np.zeros(512-SLEe)))
    for elm in range(len(LC_check[step])):
        if LC_check[step][elm] < 0.0:
            LC_check[step][elm] = 0.0
    #===========================================================================
    # Searching for local maxima (i.e., O-points)
    
    maxi1, properties = find_peaks(LC_check[step], distance = 1, rel_height=0.5, prominence=prom_lim)
    promin = properties["prominences"]
    maxi_listFP.append(maxi1)
    N_pl[:,step] = [step,len(maxi1)]

    maxi_listPROM.append(promin)
    PWidth = peak_widths(LC_check[step], maxi1, rel_height=0.5)[0]
    Width_t = np.concatenate([Width_t,PWidth])
    Pprom_t = np.concatenate([Pprom_t,promin])

    
    # #===========================================================================
print("All prominences and widths are calculated.")

timestep_check = 400
Width_t = 2*Width_t/512.

# ######## PROMINENCE DISTRIBUTION ##########
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.loglog()
prom_line = np.logspace(1.e-3,4.0, 1000)
plt.ylim(ymin=7e-1, ymax=3e3)
plt.xlim(xmin=1.5e-2,xmax=9.e-1)
nbins = 64#512
#### logbins #####
# nbinslin=np.logspace(np.min(Pprom_t),np.max(Pprom_t),nbins)
# n=plt.hist(Pprom_t, bins=np.logspace(np.log10(np.min(Pprom_t)),np.log10(np.max(Pprom_t)), nbins), log=True, alpha=0.8)
# plt.plot(1.0e-3*nbinslin[:-1], gaussian_filter(n[0],4), color='indigo')
####  log log ####
bins = np.linspace(2.e-2,8.e-1,nbins)
nprom = ax3.hist(Pprom_t,bins=bins, alpha=0.7)
# ##################
plt.plot(1.e-2*prom_line,7.e3*prom_line**-1, color='black', ls='--')
ax3.set_xlabel("Subpulse flux " + r"$(F)$", fontsize=18)
ax3.set_ylabel(r"$N_{\rm sub}$", fontsize=18)
plt.savefig("N-P-diag-loglog_from50-subst.pdf", bbox_inches="tight")

######## WIDTH-PROMINENCE 2-D DISTRIBUTION ##########
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
nbins = 64#512
h, xedges, yedges, image = plt.hist2d(Width_t,Pprom_t, bins=nbins, cmap=plt.get_cmap('gist_stern'), norm=colors.LogNorm())
########## Means and Errors #################
# mean_bin_fl = np.empty(nbins)
# std_bin_fl = np.empty(nbins)

# for bin_fl in range(nbins):
#     fl_elm = []
#     Pprom_elm = []
#     for elm in range(len(Width_t)):
#         if Width_t[elm]<=xedges[bin_fl+1]:
#             fl_elm.append(elm)
#     fl_elm=np.array(fl_elm)
#     for elem in fl_elm:
#         Pprom_elm.append(Pprom_t[elem])
#     Pprom_elm=np.array(Pprom_elm)
#     mean_bin_fl[bin_fl] = np.mean(Pprom_elm)
#     std_bin_fl[bin_fl] = stdev(Pprom_elm)

# xbins = np.empty(nbins)
# for i in range(nbins):
#     xbins[i] = (xedges[i]+xedges[i+1])/2.0

# plt.errorbar(xbins[:25], mean_bin_fl[:25], yerr=std_bin_fl[:25], ecolor = 'red')
# plt.plot(xbins[:25], mean_bin_fl[:25], 'o', markersize=2, color='black')
##############################################
muP, sigmaP = norm.fit(Pprom_t)
powerline = np.linspace(1.e-3,1.e1,1000)
plt.plot(powerline,powerline*3.6, linestyle='--', color='black', label = 'slope 1.0')
plt.vlines(np.mean(Width_t),ymin = 7.e-3, ymax = 8.e-1, label = 'mean(width)', linestyle='dotted', color='black')
plt.hlines(sigmaP,xmin = 1.e-3, xmax = 9.e-2, label = 'std.dev.(subpulse)', linestyle='dashdot', color='black')
cbar=plt.colorbar(image)
cbar.set_label('Number of subpulses', fontsize=15)
plt.ylim(ymin=7.e-3)
plt.loglog()
plt.xlabel("Subpulse width " + r"$(\delta \Phi)$", fontsize=18)
plt.ylabel("Subpulse flux " + r"$(F)$", fontsize=18)
plt.legend(loc = 'lower right')
plt.savefig("N-W-P-diag-2D-WErrorbars-loglog_from50-subst.png", bbox_inches="tight",dpi=600)

# ######## NUMBER OF PROMINENCES DISTRIBUTION ##########
# fig5 = plt.figure(5)
# ax5 = fig5.add_subplot(111)
# nbins = 40
# nmb_sub = N_pl[1][1:]/2
# mu, sigma = norm.fit(nmb_sub)
# n, bins, patches  = plt.hist(nmb_sub, bins=nbins, alpha=0.7)
# y = norm.pdf(bins, mu, sigma)
# l = plt.plot(bins, y*550, 'r--', linewidth=2)
# plt.vlines(np.mean(nmb_sub), ymin = 0, ymax = 150, color='black', linestyle='--', label='mean = %f.2' %mu)
# plt.ylim(ymax = 150)
# plt.xlabel("Number of subpulses per pulse", fontsize=18)
# plt.ylabel("N", fontsize=18)
# plt.legend(loc='upper left')
# # plt.savefig("N_subp-TRIAL_dist.png", bbox_inches="tight",dpi=100)

plt.show()