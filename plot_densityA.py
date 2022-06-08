
import numpy 
import math 
import matplotlib.pyplot as plt 
import sys 
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

def plot_density(it):

    if it=='':
       it='0'
  
    #===========================================================================
    # Parameters of the simulation

    params1=numpy.loadtxt(".././data/phys_params.dat",skiprows=1)
    params2=numpy.loadtxt(".././data/input_params.dat",skiprows=1)

    period=25796.1
    stepnmbr=int(it)
    time=stepnmbr/period
    B0=params1[0]
    rho=2.9979246e+10/params1[1]
    n0=params1[2]
    rmin=params2[10]
    rmax=params2[11]
    rpml=params2[12]
    rlc=params2[13]
    
    rmaxi=rpml/rlc
    
    #===========================================================================
    # The grid
    r=numpy.loadtxt(".././data/r.dat")
    theta=numpy.loadtxt(".././data/theta.dat")
    r=r[0:993]

    # The density
    ne_temp=numpy.loadtxt(".././data/densities/maprth_electrons_"+it+".dat")
    ni_temp=numpy.loadtxt(".././data/densities/maprth_ions_"+it+".dat")
    
    # The magnetic field
    Br_temp=numpy.loadtxt(".././data/fields/Br"+it+".dat")
    Bth_temp=numpy.loadtxt(".././data/fields/Bth"+it+".dat")
    
    ne=numpy.empty((len(r),len(theta)))
    ni=numpy.empty((len(r),len(theta)))
    Br=numpy.empty((len(r),len(theta)))
    Bth=numpy.empty((len(r),len(theta)))
    
    for ir in range(0,len(r)):
        for ith in range(0,len(theta)):
            ne[ir,ith]=ne_temp[ith,ir]/n0*(r[ir]/rmin)**2.0
            ni[ir,ith]=ni_temp[ith,ir]/n0*(r[ir]/rmin)**2.0
            Br[ir,ith]=Br_temp[ith,ir]
            Bth[ir,ith]=Bth_temp[ith,ir]
    
    #*****************************************************
    # INTERPOLATION OF THE MAPS ON A LINEAR GRID IN RADIUS
    #*****************************************************
    
    r2=numpy.empty(len(r))
    
    for ir in range(0,len(r2)):
        r2[ir]=ir/(len(r2)-1.0)*(rmax-rmin)+rmin
    
    Br2=numpy.empty((len(r2),len(theta)))
    Bth2=numpy.empty((len(r2),len(theta)))
        
    for ir in range(0,len(r2)-1):
        for ith in range(0,len(theta)-1):
            
            ri=r2[ir]
            fi=abs(ri-r)
            i=fi.argmin(axis=0)
            
            if (i==len(r)-1):
                i=i-2
                       
            rp=(ri*ri*ri-r[i]*r[i]*r[i])/(r[i+1]*r[i+1]*r[i+1]-r[i]*r[i]*r[i])
            
            f0=Br[i,ith]
            f1=Br[i+1,ith]
            
            g0=Bth[i,ith]
            g1=Bth[i+1,ith]
            
            Br2[ir,ith]=f0*(1.0-rp)+f1*rp
            Bth2[ir,ith]=g0*(1.0-rp)+g1*rp
            
    #*****************************************************
    # INTERPOLATION OF THE MAPS ON A LINEAR GRID IN RADIUS
    #*****************************************************
    
    r=r/rlc
    r2=r2/rlc
    vmin=-1.0 #-1.0
    vmax=5.0 #numpy.log10(50)
    cmap='jet' #'RdBu_r'

    #===========================================================================
    # Plot ne_wind + ni_wind
    
    fig=plt.figure(1, figsize=(8,6))
    fig.subplots_adjust(right=0.8, hspace=0.05)

    ax=plt.subplot(111,polar=True)

    def keep_center_colormap(vmi, vma, center=2):
        vmi = vmi - center
        vma = vma - center
        dv = max(-vmi, vma) * 2
        N = int(256 * dv / (vma-vmi))
        cmap_jet = cm.get_cmap(cmap, N)
        newcolors = cmap_jet(numpy.linspace(0, 1, N))
        beg = int((dv / 2 + vmi)*N / dv)
        end = N - int((dv / 2 - vma)*N / dv)
        newmap = ListedColormap(newcolors[beg:end])
        return newmap
    
    newmap = keep_center_colormap(0, 5, center=2)

    cax=ax.pcolormesh(theta,r,ne+ni,vmin=0,vmax=vmax,cmap=newmap)
#    cax=ax.pcolormesh(theta,r,numpy.log10(ne+ni),vmin=-1,vmax=2,cmap=cmap)

# Plot the limits of the zoom-in view on the layer
    cbar_axs = fig.add_axes([0.81, 0.18, 0.01, 0.65])
    cbar = fig.colorbar(cax, cax=cbar_axs)

    i1=900
    i2=750
    ax.plot(-theta[:i1],theta[:i1]*(1.-0.015*math.pi),ls='--',lw=2.0,color="red")
    ax.plot(-theta[:i2]-1.0,theta[:i2]*(1.-0.015*math.pi),ls='--',lw=2.0,color="red")
    #===========================================================================
#    plt.colorbar(cax)
    ax.plot(theta,r/r,linewidth=2.0,ls='--',color="black")
    ax.streamplot(theta,r2,Bth2,Br2,color='white',linewidth=0.4)
    ax.fill(theta,(rmin/rlc)*theta/theta,color='0.0',lw=1)
#    plt.title("at time=%0.2f" %time + r" periods, $n_e$ + $n_i$",fontsize=12)
#    ax.set_rmin(-0.0001*numpy.min(r))
    ax.set_rmax(4.0*rmaxi/7.0) #rmaxi = 10R_LC
#    ax.grid(True)

#    cbar_axs = fig.add_axes([0.81, 0.18, 0.01, 0.65])
#    cbar = fig.colorbar(cax, cax=cbar_axs)
#    cbar.set_ticks([-1, 0.2, 1.4, 2.6, 3.8, 5.0])
#    cbar.set_ticklabels([0,1,2,3,4,5])

    #===========================================================================

    #plt.figure(2)
    #plt.plot(r,ni[:,0])

#    plt.show()
    plt.savefig('./densitysumLIN_mod_%s.png' %it, bbox_inches='tight', dpi=1000)
    #===========================================================================

plot_density(sys.argv[1])
