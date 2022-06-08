import numpy as np
import pandas as pd
import math
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

wt2 = np.loadtxt('Wt-2.dat')
wt3 = np.loadtxt('Wt-3.dat')
wt4 = np.loadtxt('Wt-4.dat')
wt5 = np.loadtxt('Wt-5.dat')
wt6 = np.loadtxt('Wt-6.dat')
wt7 = np.loadtxt('Wt-7.dat')
wt8 = np.loadtxt('Wt-8.dat')
wt9 = np.loadtxt('Wt-9.dat')

# for w in wt:
#     w = w[w != 0.0]
    # nan_array = np.isnan(w)
    # not_nan_array = ~nan_array
    # w = w[not_nan_array]

wt2 = wt2[wt2 != 0.0] #change shape but remove the lines 0
wt3 = wt3[wt3 != 0.0]
wt4 = wt4[wt4 != 0.0]
wt5 = wt5[wt5 != 0.0]
wt6 = wt6[wt6 != 0.0]
wt7 = wt7[wt7 != 0.0]
wt8 = wt8[wt8 != 0.0]
wt9 = wt9[wt9 != 0.0]

##########to get rid of nan error##########
nan_array = np.isnan(wt2)
not_nan_array = ~ nan_array
wt2 = wt2[not_nan_array]
nan_array = np.isnan(wt3)
not_nan_array = ~ nan_array
wt3 = wt3[not_nan_array]
nan_array = np.isnan(wt4)
not_nan_array = ~ nan_array
wt4 = wt4[not_nan_array]
nan_array = np.isnan(wt5)
not_nan_array = ~ nan_array
wt5 = wt5[not_nan_array]
nan_array = np.isnan(wt6)
not_nan_array = ~ nan_array
wt6 = wt6[not_nan_array]
nan_array = np.isnan(wt7)
not_nan_array = ~ nan_array
wt7 = wt7[not_nan_array]
nan_array = np.isnan(wt8)
not_nan_array = ~ nan_array
wt8 = wt8[not_nan_array]
nan_array = np.isnan(wt9)
not_nan_array = ~ nan_array
wt9 = wt9[not_nan_array]

nbinslin2=np.logspace(1.e-2,2,100)
nbinslin3=np.logspace(7.e-2,2,100)
nbinslin4=np.logspace(1.e-2,1.5,100)
nbinslin5=np.logspace(1.e-2,1.4,100)
nbinslin6=np.logspace(1.e-2,1.3,100)
nbinslin7=np.logspace(1.e-2,1.15,100)
nbinslin8=np.logspace(1.e-2,1.2,100)
nbinslin9=np.logspace(1.e-2,1.05,100)
nbins=100

# Setting up a colormap that's a simple transtion
n_lines = 8
x = np.linspace(0, 10, 200)
y = np.sin(x[:, None] + np.pi * np.linspace(0, 1, n_lines))
c = np.arange(1, n_lines + 1)

cmap = mp.cm.get_cmap('jet', n_lines)

fig = plt.figure(1)
ax=plt.subplot(111)

dummie_cax = ax.scatter(c, c, c=c, cmap=cmap)

# Clear axis
ax.cla()

n2=plt.hist(wt2, bins=np.logspace(np.log10(np.min(wt2)),np.log10(np.max(wt2)), nbins), log=True, alpha=0)
plt.plot(0.8e-2*nbinslin2[:-1], n2[0], label='1-2', color='darkblue')
n3=plt.hist(wt3, bins=np.logspace(np.log10(np.min(wt3)),np.log10(np.max(wt3)), nbins), log=True, alpha=0)
plt.plot(1.2e-2*nbinslin3[:-1], n3[0], label='2-3', color='blue')
n4=plt.hist(wt4, bins=np.logspace(np.log10(np.min(wt4)),np.log10(np.max(wt4)), nbins), log=True, alpha=0)
plt.plot(2.5e-2*nbinslin4[:-1], n4[0], label='3-4', color='deepskyblue')
n5=plt.hist(wt5, bins=np.logspace(np.log10(np.min(wt5)),np.log10(np.max(wt5)), nbins), log=True, alpha=0)
plt.plot(3.7e-2*nbinslin5[:-1], n5[0], label='4-5', color='lightgreen')
n6=plt.hist(wt6, bins=np.logspace(np.log10(np.min(wt6)),np.log10(np.max(wt6)), nbins), log=True, alpha=0)
plt.plot(5.e-2*nbinslin6[:-1], n6[0], label='5-6', color='lime')
n7=plt.hist(wt7, bins=np.logspace(np.log10(np.min(wt7)),np.log10(np.max(wt7)), nbins), log=True, alpha=0)
plt.plot(6.9e-2*nbinslin7[:-1], n7[0], label='6-7', color='orange')
n8=plt.hist(wt8, bins=np.logspace(np.log10(np.min(wt8)),np.log10(np.max(wt8)), nbins), log=True, alpha=0)
plt.plot(6.3e-2*nbinslin8[:-1], n8[0], label='7-8', color='orangered')
n9=plt.hist(wt9, bins=np.logspace(np.log10(np.min(wt9)),np.log10(np.max(wt9)), nbins), log=True, alpha=0, label='8-9', color=(1.0, 0.0, 0.0))
plt.plot(9.e-2*nbinslin9[:-1], n9[0], label='8-9', color='maroon')

cbar = fig.colorbar(dummie_cax, ticks=c)
cbar.set_ticks([1,1.85,2.74,3.62,4.5,5.38,6.26,7.14,8.02])
cbar.set_ticklabels([1,2,3,4,5,6,7,8,9])

lnspc = np.linspace(np.min(wt2), np.max(wt2), len(wt2))
plt.plot(lnspc, 1.2e2*lnspc**-1.0, linestyle='dashed', color='black')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"$w/R_{\rm LC}$", fontsize=18)
ax.set_ylabel(r"$N$", fontsize=18)
ax.text(5.e-2, 2.5e3, r'$N(w) \propto w^{-1}$', fontsize=13)
ax.text(1.34,7.8, r'$R_{\rm LC}$', fontsize=10)

ax.set_ylim(ymin=10.0, ymax=1.e4)
ax.set_xlim(xmin=5.e-3, xmax=1)
plt.xlim(xmin=5.e-3,xmax=1.0)
# #===========================================================================
plt.show()
# plt.savefig('size_hist-r_we-C.pdf', bbox_inches='tight')
        
# #===========================================================================