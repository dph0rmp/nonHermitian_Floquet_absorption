# -*- coding: utf-8 -*-
"""

Non-Hermitian Floquet dynamics in absorption spectroscopy

R M Potvliege

This code calculates the results presented in figure 1 of the paper and
plots the figure. 

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


twopi = 2.0*np.pi
zi = complex(0.0,1.0)

Gamma_1 =  complex(twopi,0.0)*3.6
Gamma_2 =  complex(twopi,0.0)*0.9
Delta_p1 = complex(twopi,0.0)*2.0/5.0
Delta_p2 = complex(twopi,0.0)*5.0/5.0
Delta_p3 = complex(twopi,0.0)*(-1.0)/5.0
Delta_p4 = complex(twopi,0.0)*4.0/5.0
Omega_01 = complex(twopi,0.0)*10.0
Omega_02 = complex(twopi,0.0)*5.0
Omega_13 = complex(twopi,0.0)*9.0
Omega_23 = complex(twopi,0.0)*11.0
Omega_14 = complex(twopi,0.0)*6.0
Omega_24 = complex(twopi,0.0)*9.0
omega =    complex(twopi,0.0)*1.0


tmax = 20.0

ndim = 5

dcdt = np.zeros(ndim)*complex(1.0,0.0)
hmat = np.zeros((ndim,ndim))*complex(1.0,0.0)
hmat0 = np.zeros((ndim,ndim))*complex(1.0,0.0)
vmatpiom = np.zeros((ndim,ndim))*complex(1.0,0.0)
vmatmiom = np.zeros((ndim,ndim))*complex(1.0,0.0)
rho = np.zeros((ndim,ndim))*complex(1.0,0.0)
rhodot = np.zeros((ndim,ndim))*complex(1.0,0.0)
drhodt = np.zeros(ndim*ndim)*complex(1.0,0.0)

# Here we assume that the Rabi frequencies are real...

hmat0[0,1] = -0.5*Omega_01
hmat0[0,2] = -0.5*Omega_02
hmat0[1,1] = -Delta_p1
vmatpiom[1,3] = -0.5*Omega_13
vmatmiom[1,3] = -0.5*Omega_13
vmatpiom[1,4] = -0.5*Omega_14
vmatmiom[1,4] = -0.5*Omega_14
hmat0[2,2] = -Delta_p2
vmatpiom[2,3] = -0.5*Omega_23
vmatmiom[2,3] = -0.5*Omega_23
vmatpiom[2,4] = -0.5*Omega_24
vmatmiom[2,4] = -0.5*Omega_24
hmat0[3,3] = -Delta_p3
hmat0[4,4] = -Delta_p4

for j in range(0, ndim-1):
    for i in range(j+1, ndim):
        hmat0[i,j] = np.conj(hmat0[j,i])
        vmatpiom[i,j] = np.conj(vmatmiom[j,i])
        vmatmiom[i,j] = np.conj(vmatpiom[j,i])


nindi = []
nindj = []
for j in range(0, ndim):
    for i in range(0, ndim):
      nindi.append(i)
      nindj.append(j)


rhov = np.zeros(ndim*ndim)*complex(1.0,0.0)
rhovdot = np.zeros(ndim*ndim)*complex(1.0,0.0)
rhovdotLplus = np.zeros(ndim*ndim)*complex(1.0,0.0)
rhovdotLminus = np.zeros(ndim*ndim)*complex(1.0,0.0)
lmat = np.zeros((ndim*ndim,ndim*ndim))*complex(1.0,0.0)
lmat0 = np.zeros((ndim*ndim,ndim*ndim))*complex(1.0,0.0)
lmatplus = np.zeros((ndim*ndim,ndim*ndim))*complex(1.0,0.0)
lmatminus = np.zeros((ndim*ndim,ndim*ndim))*complex(1.0,0.0)

drhodt_Lpl = np.zeros(ndim*ndim)*complex(1.0,0.0)
drhodt_Lmn = np.zeros(ndim*ndim)*complex(1.0,0.0)

def rhofun0(rhov):
    for k in range(0, ndim*ndim):
        rho[nindi[k],nindj[k]] = rhov[k]
    hmat = hmat0
    rhodot = (hmat.dot(rho)-rho.dot(hmat))/zi
    rhodot[0,0] = rhodot[0,0] + rho[1,1]*Gamma_1 + rho[2,2]*Gamma_2
    rhodot[1,1] = rhodot[1,1] - rho[1,1]*Gamma_1 
    rhodot[2,2] = rhodot[2,2] - rho[2,2]*Gamma_2
    rhodot[0,1] = rhodot[0,1] - rho[0,1]*Gamma_1/2.0
    rhodot[1,0] = rhodot[1,0] - rho[1,0]*Gamma_1/2.0
    rhodot[0,2] = rhodot[0,2] - rho[0,2]*Gamma_2/2.0
    rhodot[2,0] = rhodot[2,0] - rho[2,0]*Gamma_2/2.0
    rhodot[1,2] = rhodot[1,2] - rho[1,2]*(Gamma_1+Gamma_2)/2.0
    rhodot[2,1] = rhodot[2,1] - rho[2,1]*(Gamma_1+Gamma_2)/2.0
    rhodot[1,3] = rhodot[1,3] - rho[1,3]*Gamma_1/2.0
    rhodot[3,1] = rhodot[3,1] - rho[3,1]*Gamma_1/2.0
    rhodot[1,4] = rhodot[1,4] - rho[1,4]*Gamma_1/2.0
    rhodot[4,1] = rhodot[4,1] - rho[4,1]*Gamma_1/2.0
    rhodot[2,3] = rhodot[2,3] - rho[2,3]*Gamma_2/2.0
    rhodot[3,2] = rhodot[3,2] - rho[3,2]*Gamma_2/2.0
    rhodot[2,4] = rhodot[2,4] - rho[2,4]*Gamma_2/2.0
    rhodot[4,2] = rhodot[4,2] - rho[4,2]*Gamma_2/2.0
    for k in range(0, ndim*ndim):
         drhodt[k] = rhodot[nindi[k],nindj[k]]
    return drhodt

def rhofunLplus(rhov_in):
    for k in range(0, ndim*ndim):
        rho[nindi[k],nindj[k]] = rhov_in[k]
    hmat = vmatmiom
    rhodot = (hmat.dot(rho)-rho.dot(hmat))/zi
    for k in range(0, ndim*ndim):
         drhodt_Lpl[k] = rhodot[nindi[k],nindj[k]]
    return drhodt_Lpl

def rhofunLminus(rhov_in):
    for k in range(0, ndim*ndim):
        rho[nindi[k],nindj[k]] = rhov_in[k]
    hmat = vmatpiom
    rhodot = (hmat.dot(rho)-rho.dot(hmat))/zi
    for k in range(0, ndim*ndim):
         drhodt_Lmn[k] = rhodot[nindi[k],nindj[k]]
    return drhodt_Lmn

for k in range(0, ndim*ndim):
    rhov =  np.zeros(ndim*ndim)*complex(1.0,0.0)
    rhov[k] = complex(1.0,0.0)
    rhovLpl = rhov
    rhovLmn = rhov
    rhovdot = rhofun0(rhov)
    rhovdotLplus = rhofunLplus(rhovLpl)
    rhovdotLminus = rhofunLminus(rhovLmn)
    for l in range(0, ndim*ndim):
        lmat0[l,k] = rhovdot[l]
        lmatplus[l,k] = rhovdotLplus[l]
        lmatminus[l,k] = rhovdotLminus[l]     
     

###################################
#
#  Floquet...
#
###################################

nnmax = 50
nnmin = -nnmax

ndimFl = (nnmax-nnmin+1)*ndim*ndim
lmatFl = np.zeros((ndimFl,ndimFl))*complex(1.0,0.0)
rFl = np.zeros(ndimFl)*complex(1.0,0.0)
drFldt = np.zeros(ndimFl)*complex(1.0,0.0)

indFl00 = [0 for i in range(nnmin, nnmax+1)]
indFl01 = [0 for i in range(nnmin, nnmax+1)]
indFl02 = [0 for i in range(nnmin, nnmax+1)]
jFl = -1
for nnj in range(nnmin, nnmax+1):
    for j in range(0, ndim*ndim):
        jFl = jFl + 1
        if j == 0:
            indFl00[nnj] = jFl
        elif j == 5:
            indFl01[nnj] = jFl
        elif j == 10:
            indFl02[nnj] = jFl
        if nnj == 0 and j == 0:
            jFl0 = jFl
        iFl = -1
        for nni in range(nnmin,nnmax+1):
            for i in range(0, ndim*ndim):
                iFl = iFl + 1
                if nni == nnj:
                    lmatFl[iFl,jFl] = lmat0[i,j]
                    if iFl == jFl:
                        lmatFl[iFl,jFl] = lmatFl[iFl,jFl] + nni*omega*zi
                elif nni == nnj-1:
                    lmatFl[iFl,jFl] = lmatminus[i,j]
                elif nni == nnj+1:
                    lmatFl[iFl,jFl] = lmatplus[i,j]

def rhoFlfunL(t, rhovFl):
    drFldt = lmatFl.dot(rhovFl)
    return drFldt                    


rFlinit = np.zeros(ndimFl)*complex(1.0,0.0)
rFlinit[jFl0] = 1.0

ntimes = 2001

solFl = solve_ivp(rhoFlfunL, [0.0,tmax], rFlinit, method='RK45',
                  t_eval=np.linspace(0.0,tmax,ntimes),
                  atol=1.e-13, rtol=1.e-13)


time = np.zeros(ntimes)
r00 = np.zeros(ntimes)
imr01 = np.zeros(ntimes)
imr02 = np.zeros(ntimes)
for it in range(0,ntimes):
    tit = solFl.t[it]
    yFl00 = complex(0.0,0.0)
    yFl01 = complex(0.0,0.0)
    yFl02 = complex(0.0,0.0)
    for nn in range(nnmin, nnmax+1):
        yFl00 = yFl00 + solFl.y[indFl00[nn]][it]*np.exp(-nn*zi*omega*tit)
        yFl01 = yFl01 + solFl.y[indFl01[nn]][it]*np.exp(-nn*zi*omega*tit)
        yFl02 = yFl02 + solFl.y[indFl02[nn]][it]*np.exp(-nn*zi*omega*tit)
    time[it] = tit
    r00[it] = yFl00.real
    imr01[it] = yFl01.imag
    imr02[it] = yFl02.imag


r00_0 = np.zeros(ntimes)
r00_2 = np.zeros(ntimes)
r00_4 = np.zeros(ntimes)
imr01_0 = np.zeros(ntimes)
imr01_2 = np.zeros(ntimes)
imr01_4 = np.zeros(ntimes)
imr02_0 = np.zeros(ntimes)
imr02_2 = np.zeros(ntimes)
imr02_4 = np.zeros(ntimes)
for it in range(0,ntimes):
    r00_0[it] = solFl.y[indFl00[0]][it].real
    r00_2[it] = solFl.y[indFl00[2]][it].real
    r00_4[it] = solFl.y[indFl00[4]][it].real
    imr01_0[it] = solFl.y[indFl01[0]][it].imag
    imr01_2[it] = solFl.y[indFl01[2]][it].imag
    imr01_4[it] = solFl.y[indFl01[4]][it].imag
    imr02_0[it] = solFl.y[indFl02[0]][it].imag
    imr02_2[it] = solFl.y[indFl02[2]][it].imag
    imr02_4[it] = solFl.y[indFl02[4]][it].imag

##################################################
# Results without coupling field:
    
def rhofun_probeonly(t, rhov):
    for k in range(0, ndim*ndim):
        rho[nindi[k],nindj[k]] = rhov[k]
    #epiom = np.exp(zi*omega*t)
    #emiom = np.conj(epiom)
    #hmat = hmat0 + vmatmiom*emiom + vmatpiom*epiom
    hmat = hmat0
    rhodot = (hmat.dot(rho)-rho.dot(hmat))/zi
    rhodot[0,0] = rhodot[0,0] + rho[1,1]*Gamma_1 + rho[2,2]*Gamma_2
    rhodot[1,1] = rhodot[1,1] - rho[1,1]*Gamma_1 
    rhodot[2,2] = rhodot[2,2] - rho[2,2]*Gamma_2
    rhodot[0,1] = rhodot[0,1] - rho[0,1]*Gamma_1/2.0
    rhodot[1,0] = rhodot[1,0] - rho[1,0]*Gamma_1/2.0
    rhodot[0,2] = rhodot[0,2] - rho[0,2]*Gamma_2/2.0
    rhodot[2,0] = rhodot[2,0] - rho[2,0]*Gamma_2/2.0
    rhodot[1,2] = rhodot[1,2] - rho[1,2]*(Gamma_1+Gamma_2)/2.0
    rhodot[2,1] = rhodot[2,1] - rho[2,1]*(Gamma_1+Gamma_2)/2.0
    rhodot[1,3] = rhodot[1,3] - rho[1,3]*Gamma_1/2.0
    rhodot[3,1] = rhodot[3,1] - rho[3,1]*Gamma_1/2.0
    rhodot[1,4] = rhodot[1,4] - rho[1,4]*Gamma_1/2.0
    rhodot[4,1] = rhodot[4,1] - rho[4,1]*Gamma_1/2.0
    rhodot[2,3] = rhodot[2,3] - rho[2,3]*Gamma_2/2.0
    rhodot[3,2] = rhodot[3,2] - rho[3,2]*Gamma_2/2.0
    rhodot[2,4] = rhodot[2,4] - rho[2,4]*Gamma_2/2.0
    rhodot[4,2] = rhodot[4,2] - rho[4,2]*Gamma_2/2.0
    for k in range(0, ndim*ndim):
         drhodt[k] = rhodot[nindi[k],nindj[k]]
    return drhodt


rhoprobeonly_init = np.zeros(ndim*ndim)*complex(1.0,0.0)
rhoprobeonly_init[0] = 1.0
sol = solve_ivp(rhofun_probeonly, [0.0,tmax], rhoprobeonly_init, method='RK45',
                t_eval=np.linspace(0.0,tmax,ntimes),
                atol=1.e-13, rtol=1.e-13)

rhoprobeonly00 = sol.y[0].real

##################################################
# Results with coupling field, direct integration of the Lindblad equation,
# Eq. 18 of the paper:
    
def rhofun(t, rhov):
    for k in range(0, ndim*ndim):
        rho[nindi[k],nindj[k]] = rhov[k]
    epiom = np.exp(zi*omega*t)
    emiom = np.conj(epiom)
    hmat = hmat0 + vmatmiom*emiom + vmatpiom*epiom
    rhodot = (hmat.dot(rho)-rho.dot(hmat))/zi
    rhodot[0,0] = rhodot[0,0] + rho[1,1]*Gamma_1 + rho[2,2]*Gamma_2
    rhodot[1,1] = rhodot[1,1] - rho[1,1]*Gamma_1 
    rhodot[2,2] = rhodot[2,2] - rho[2,2]*Gamma_2
    rhodot[0,1] = rhodot[0,1] - rho[0,1]*Gamma_1/2.0
    rhodot[1,0] = rhodot[1,0] - rho[1,0]*Gamma_1/2.0
    rhodot[0,2] = rhodot[0,2] - rho[0,2]*Gamma_2/2.0
    rhodot[2,0] = rhodot[2,0] - rho[2,0]*Gamma_2/2.0
    rhodot[1,2] = rhodot[1,2] - rho[1,2]*(Gamma_1+Gamma_2)/2.0
    rhodot[2,1] = rhodot[2,1] - rho[2,1]*(Gamma_1+Gamma_2)/2.0
    rhodot[1,3] = rhodot[1,3] - rho[1,3]*Gamma_1/2.0
    rhodot[3,1] = rhodot[3,1] - rho[3,1]*Gamma_1/2.0
    rhodot[1,4] = rhodot[1,4] - rho[1,4]*Gamma_1/2.0
    rhodot[4,1] = rhodot[4,1] - rho[4,1]*Gamma_1/2.0
    rhodot[2,3] = rhodot[2,3] - rho[2,3]*Gamma_2/2.0
    rhodot[3,2] = rhodot[3,2] - rho[3,2]*Gamma_2/2.0
    rhodot[2,4] = rhodot[2,4] - rho[2,4]*Gamma_2/2.0
    rhodot[4,2] = rhodot[4,2] - rho[4,2]*Gamma_2/2.0
    
    
    for k in range(0, ndim*ndim):
         drhodt[k] = rhodot[nindi[k],nindj[k]]
    return drhodt


# Uncomment the following statements for printing out the values of
# rho_00 etc.at t = tmax.

##rho_init = np.zeros(ndim*ndim)*complex(1.0,0.0)
##rho_init[0] = 1.0
##sol = solve_ivp(rhofun, [0.0,tmax], rho_init, method='RK45',
##                t_eval=np.linspace(0.0,tmax,ntimes),
##                atol=1.e-13, rtol=1.e-13)
##
###  Print out rho_00(t), Im rho_10(t) and Im rho_20(t):
##print(sol.t[ntimes-1],sol.y[0][ntimes-1])   # rho_00 Lindblad
##print(time[ntimes-1],r00[ntimes-1])         # rho_00 Floquet
##print(sol.t[ntimes-1],-sol.y[5][ntimes-1])  # Im rho_10 Lindblad
##print(time[ntimes-1],-imr01[ntimes-1])      # Im rho_10 Floquet
##print(sol.t[ntimes-1],-sol.y[10][ntimes-1]) # Im rho_20 Lindblad
##print(time[ntimes-1],-imr02[ntimes-1])      # Im rho_20 Floquet
##import sys
##sys.exit()



####################################################
# Plot the results
 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'

fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(9.4,6.0))
ax0 = axs[0,0]
ax1 = axs[0,1]
ax2 = axs[0,2]
ax3 = axs[1,0]
ax4 = axs[1,1]
ax5 = axs[1,2]
 
ax0.set_xlim([-0.10,1.05])
ax0.set_ylim([0,1])
sc = 0.5
y0 = 0.1
y1 = 0.8-0.2*sc
y2 = 0.8-0.5*sc
y3 = 0.8+0.1*sc
y4 = 0.8-0.4*sc
ax0.plot([0.45,0.55],[y0,y0], color='k')
ax0.plot([0.15,0.25],[y1,y1], color='k')
ax0.plot([0.75,0.85],[y2,y2], color='k')
ax0.plot([0.45,0.55],[y3,y3], color='k')
ax0.plot([0.45,0.55],[y4,y4], color='k')
ax0.annotate(s='', xy=(0.20,y1), xytext=(0.5,0.1), arrowprops=dict(arrowstyle='<->', color='r', shrinkA=4, shrinkB=4))
ax0.annotate(s='', xy=(0.80,y2), xytext=(0.5,0.1), arrowprops=dict(arrowstyle='<->', color='r', shrinkA=4, shrinkB=4))
ax0.annotate(s='', xy=(0.50,y3), xytext=(0.20,y1), arrowprops=dict(arrowstyle='<->', color='grey', shrinkA=4, shrinkB=4))
ax0.annotate(s='', xy=(0.50,y4), xytext=(0.20,y1), arrowprops=dict(arrowstyle='<->', color='grey', shrinkA=10, shrinkB=10))
ax0.annotate(s='', xy=(0.50,y3), xytext=(0.80,y2), arrowprops=dict(arrowstyle='<->', color='grey', shrinkA=4, shrinkB=4))
ax0.annotate(s='', xy=(0.50,y4), xytext=(0.80,y2), arrowprops=dict(arrowstyle='<->', color='grey', shrinkA=10, shrinkB=10))
ax0.text(0.42,1.00,'${(a)}$',fontsize =15)
ax0.text(0.40-0.03,0.10-0.025,'$0$',fontsize =15)
ax0.text(0.10-0.03,y1-0.025,'$1$',fontsize =15)
ax0.text(0.90-0.015,y2-0.025,'$2$',fontsize =15)
ax0.text(0.60-0.015,y3-0.025,'$3$',fontsize =15)
ax0.text(0.50-0.03,y4-0.080,'$4$',fontsize =15)
ax0.axis('off')

ax1.set_ylim([-0.30,1.05])
ax1.plot(time[0:200],rhoprobeonly00[0:200], color='#1f77b4', linestyle='dotted')
ax1.plot(time[0:200],r00[0:200])
ax1.plot(time[0:200],-imr01[0:200])
ax1.plot(time[0:200],-imr02[0:200])
ax1.xaxis.set_minor_locator(MultipleLocator(0.2))
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
ax1.tick_params(direction='in',bottom=True,top=False,left=True,right=False)
ax1.tick_params(which='minor',direction='in',bottom=True,top=False,left=True,right=False)
ax1.set_xlabel(r'$t$ $\mathrm{(}\mathrm{ns)}$', fontsize=15)
ax1.text(0.85,1.05,'${(b)}$',fontsize =15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.set_ylim([-0.30,1.05])
ax2.plot(time,rhoprobeonly00, color='#1f77b4', linestyle='dotted')
ax2.plot(time,r00)
ax2.plot(time,-imr01)
ax2.plot(time,-imr02)
ax2.xaxis.set_minor_locator(MultipleLocator(2.0))
ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
ax2.tick_params(direction='in',bottom=True,top=False,left=True,right=False)
ax2.tick_params(which='minor',direction='in',bottom=True,top=False,left=True,right=False)
ax2.set_xlabel(r'$t$ $\mathrm{(}\mathrm{ns)}$', fontsize=15)
ax2.text(8.5,1.05,'${(c)}$',fontsize =15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax3.set_yscale('log')
ax3.set_ylim([0.0007,1.6])
ax3.plot(time[0:600],abs(r00_0[0:600]), color='#1f77b4', linestyle='solid')
ax3.plot(time[0:600],abs(r00_2[0:600]), color='#1f77b4', linestyle=(0,(10,3)))
ax3.plot(time[0:600],abs(r00_4[0:600]), color='#1f77b4', linestyle='dashed')
ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
ax3.tick_params(direction='in',bottom=True,top=False,left=True,right=False)
ax3.tick_params(which='minor',direction='in',bottom=True,top=False,left=True,right=False)
ax3.set_xlabel(r'$t$ $\mathrm{(}\mathrm{ns)}$', fontsize=15)
ax3.text(2.55,2.0,'${(d)}$',fontsize =15)
ax3.text(4.50,0.33,'$N=0$',fontsize =15)
ax3.text(3.6,0.033,'$N=2$',fontsize =15)
ax3.text(4.50,0.004,'$N=4$',fontsize =15)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax4.set_yscale('log')
ax4.set_ylim([0.0007,1.6])
ax4.plot(time[0:600],abs(imr01_0[0:600]), color='#ff7f0e', linestyle='solid')
ax4.plot(time[0:600],abs(imr01_2[0:600]), color='#ff7f0e', linestyle=(0,(10,3)))
ax4.plot(time[0:600],abs(imr01_4[0:600]), color='#ff7f0e', linestyle='dashed')
ax4.xaxis.set_minor_locator(MultipleLocator(0.5))
ax4.tick_params(direction='in',bottom=True,top=False,left=True,right=False)
ax4.tick_params(which='minor',direction='in',bottom=True,top=False,left=True,right=False)
ax4.set_xlabel(r'$t$ $\mathrm{(}\mathrm{ns)}$', fontsize=15)
ax4.text(2.55,2.0,'${(e)}$',fontsize =15)
ax4.text(4.50,0.15/1.2,'$N=0$',fontsize =15)
ax4.text(3.60,0.030/1.2,'$N=2$',fontsize =15)
ax4.text(4.50,0.0055/1.2,'$N=4$',fontsize =15)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

ax5.set_yscale('log')
ax5.set_ylim([0.0007,1.6])
ax5.plot(time[0:600],abs(imr02_0[0:600]), color='#2ca02c', linestyle='solid')
ax5.plot(time[0:600],abs(imr02_2[0:600]), color='#2ca02c', linestyle=(0,(10,3)))
ax5.plot(time[0:600],abs(imr02_4[0:600]), color='#2ca02c', linestyle='dashed')
ax5.xaxis.set_minor_locator(MultipleLocator(0.5))
ax5.tick_params(direction='in',bottom=True,top=False,left=True,right=False)
ax5.tick_params(which='minor',direction='in',bottom=True,top=False,left=True,right=False)
ax5.set_xlabel(r'$t$ $\mathrm{(}\mathrm{ns)}$', fontsize=15)
ax5.text(2.55,2.0,'${(f)}$',fontsize =15)
ax5.text(4.50,0.08/1.2,'$N=0$',fontsize =15)
ax5.text(4.50,0.008/1.2,'$N=4$',fontsize =15)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)


fig.tight_layout(pad=0.9)
plt.savefig('figure1.pdf', format='pdf')

plt.show()

                                             

