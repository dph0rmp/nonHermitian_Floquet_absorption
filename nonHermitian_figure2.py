# -*- coding: utf-8 -*-
"""

Non-Hermitian Floquet dynamics in absorption spectroscopy

R M Potvliege

This code calculates the results presented in figure 2 of the paper and
plots the figure. 

"""

import numpy as np
from scipy.linalg import solve, eig
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


twopi = 2.0*np.pi
zi = complex(0.0,1.0)

dipmom_1 = complex(1.0,0.0)*2.0
dipmom_2 = complex(1.0,0.0)*1.0


Gamma_1 =  complex(twopi,0.0)*3.6
Gamma_2 =  complex(twopi,0.0)*0.9
Delta_p1 = complex(twopi,0.0)*2.0/5.0
Delta_p2 = complex(twopi,0.0)*5.0/5.0
Delta_p3 = complex(twopi,0.0)*(-1.0)/5.0
Delta_p4 = complex(twopi,0.0)*4.0/5.0
Omega_01 = complex(twopi,0.0)*1.0*10.0
Omega_02 = complex(twopi,0.0)*0.5*10.0
Omega_13 = complex(twopi,0.0)*9.0
Omega_23 = complex(twopi,0.0)*11.0
Omega_14 = complex(twopi,0.0)*6.0
Omega_24 = complex(twopi,0.0)*9.0
omega =    complex(twopi,0.0)*1.0

omega_p =  complex(twopi,0.0)*100.0

tmax = 200.0

ndimA = 1
ndimB = 4
ndim = ndimA + ndimB

dcdt = np.zeros(ndim)*complex(1.0,0.0)
hmat = np.zeros((ndim,ndim))*complex(1.0,0.0)
hmat0 = np.zeros((ndim,ndim))*complex(1.0,0.0)
vmatpiom = np.zeros((ndim,ndim))*complex(1.0,0.0)
vmatmiom = np.zeros((ndim,ndim))*complex(1.0,0.0)
rho = np.zeros((ndim,ndim))*complex(1.0,0.0)
rhodot = np.zeros((ndim,ndim))*complex(1.0,0.0)
drhodt = np.zeros(ndim*ndim)*complex(1.0,0.0)

# Here we assume that the Rabi frequencies are real...

def build_hamiltonian():
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


nnmax = 30
nnmin = -nnmax
    
ndimFl = (nnmax-nnmin+1)*ndim*ndim
lmatFl = np.zeros((ndimFl,ndimFl))*complex(1.0,0.0)
rFl = np.zeros(ndimFl)*complex(1.0,0.0)
drFldt = np.zeros(ndimFl)*complex(1.0,0.0)
    
indFl00 = [0 for i in range(nnmin, nnmax+1)]
indFl01 = [0 for i in range(nnmin, nnmax+1)]
indFl02 = [0 for i in range(nnmin, nnmax+1)]
indFl10 = [0 for i in range(nnmin, nnmax+1)]
indFl20 = [0 for i in range(nnmin, nnmax+1)]

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

def Floquet_noweakprobe():
    build_hamiltonian()
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
    return jFl0
                        

def rhoFlfunL(t, rhovFl):
    drFldt = lmatFl.dot(rhovFl)
    return drFldt                    


def Floquet_weakprobe(omega_p): 
    global bvec, LHSmat, Fmat      
    jFl = -1
    for nnj in range(nnmin, nnmax+1):
        for l in range(ndimA, ndim):
            jFl = jFl + 1
            if nnj == 0:
                if l == 1:
                    bvec[jFl] = -Omega_01/2.0
                elif l == 2:
                    bvec[jFl] = -Omega_02/2.0
            iFl = -1
            for nni in range(nnmin,nnmax+1):
                for i in range(ndimA, ndim):
                    iFl = iFl + 1
                    if i == 1:
                        indFl10[nni] = iFl
                    elif i == 2:
                        indFl20[nni] = iFl
                    if iFl == jFl:
                        if i == 1:
                            LHSmat[iFl,jFl] = Delta_p1 + zi*Gamma_1/2.0 + nni*omega
                           
                        elif i == 2:
                            LHSmat[iFl,jFl] = Delta_p2 + zi*Gamma_2/2.0 + nni*omega
                            
                        elif i == 3:
                            LHSmat[iFl,jFl] = Delta_p3 + nni*omega
                        elif i == 4:
                            LHSmat[iFl,jFl] = Delta_p4 + nni*omega
                        Fmat[iFl,jFl] = LHSmat[iFl,jFl] - omega_p
                    elif nni == nnj-1:
                        LHSmat[iFl,jFl] = OmegaB[i,l]/2.0
                        Fmat[iFl,jFl] = LHSmat[iFl,jFl]
                    elif nni == nnj+1:
                        LHSmat[iFl,jFl] = OmegaB[i,l]/2.0
                        Fmat[iFl,jFl] = LHSmat[iFl,jFl]

def Floquet_weakprobe_eigen(omega_p):
    global eigenwkprb, uwkprb, vwkprb
    Floquet_weakprobe(omega_p)
    reswkprb = eig(Fmat,left=True,right=True)
    eigenwkprb = reswkprb[0]
    vwkprb = reswkprb[2]
    uwkprb = np.conj(reswkprb[1].transpose())
    for k in range(0,nnFlB):
        factor = uwkprb[k,:].dot(vwkprb[:,k])
        uwkprb[k,:] = uwkprb[k,:]/factor
    

OmegaB = np.zeros((ndim,ndim))*complex(1.0,0.0)
OmegaB[1,3] = Omega_13
OmegaB[1,4] = Omega_14
OmegaB[2,3] = Omega_23
OmegaB[2,4] = Omega_24
for j in range(ndimA, ndim-1):
    for i in range(j+1, ndim):
        OmegaB[i,j] = np.conj(OmegaB[j,i])

nnFlB = (nnmax-nnmin+1)*ndimB
Fmat = np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
LHSmat = np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
bvec = np.zeros(nnFlB)*complex(1.0,0.0)
xvec = np.zeros(nnFlB)*complex(1.0,0.0)

numbalpha = 31
alpha = np.zeros(numbalpha)
r00_0 = np.zeros(numbalpha)
r00_2 = np.zeros(numbalpha)
r00_4 = np.zeros(numbalpha)
imr01_0 = np.zeros(numbalpha)
imr01_2 = np.zeros(numbalpha)
imr01_4 = np.zeros(numbalpha)
imr02_0 = np.zeros(numbalpha)
imr02_2 = np.zeros(numbalpha)
imr02_4 = np.zeros(numbalpha)
imr01wkprb_0 = np.zeros(numbalpha)
imr02wkprb_0 = np.zeros(numbalpha)
for ka in range(0,numbalpha):
    alpha[ka] = (10.0**(ka/10.0))
    Omega_01 = complex(twopi,0.0)*1.0*10.0/alpha[ka]
    Omega_02 = complex(twopi,0.0)*0.5*10.0/alpha[ka]

    jFl0 = Floquet_noweakprobe()

    rFlinit = np.zeros(ndimFl)*complex(1.0,0.0)
    rFlinit[jFl0] = 1.0

    res = eig(lmatFl,left=True,right=True)
    eigenv = res[0]
    c = res[2]
    d = np.conj(res[1])


    coeffinit = np.zeros(ndimFl)*complex(1.0,0.0)
    for k in range(0,ndimFl):
        coeffinit[k] = d[:,k].dot(rFlinit)/d[:,k].dot(c[:,k])

    rprop = np.zeros(ndimFl)*complex(1.0,0.0)
    for k in range(0,ndimFl):
        rprop = rprop + coeffinit[k]*np.exp(eigenv[k]*tmax)*c[:,k]   
 

    r00_0[ka] = rprop[indFl00[0]].real
    r00_2[ka] = rprop[indFl00[2]].real
    r00_4[ka] = rprop[indFl00[4]].real
    imr01_0[ka] = rprop[indFl01[0]].imag
    imr01_2[ka] = rprop[indFl01[2]].imag
    imr01_4[ka] = rprop[indFl01[4]].imag
    imr02_0[ka] = rprop[indFl02[0]].imag
    imr02_2[ka] = rprop[indFl02[2]].imag
    imr02_4[ka] = rprop[indFl02[4]].imag
    
    Floquet_weakprobe(omega_p)
    x = solve(LHSmat,bvec)
    imr01wkprb_0[ka] = -x[indFl10[0]].imag
    imr02wkprb_0[ka] = -x[indFl20[0]].imag

# Reduces the strength of the probe field:

Omega_01 = complex(twopi,0.0)*10.0/10000.0
Omega_02 = complex(twopi,0.0)*5.0/10000.0


####################################################
# Part (b) of the figure: without the coupling field

btestf2 = 1.e-15  # Factor used to turn the coupling field off

OmegaB = np.zeros((ndim,ndim))*complex(1.0,0.0)
OmegaB[1,3] = Omega_13*btestf2
OmegaB[1,4] = Omega_14*btestf2
OmegaB[2,3] = Omega_23*btestf2
OmegaB[2,4] = Omega_24*btestf2
for j in range(ndimA, ndim-1):
    for i in range(j+1, ndim):
        OmegaB[i,j] = np.conj(OmegaB[j,i])

eigenwkprb = np.zeros(nnFlB)*complex(1.0,0.0)
uwkprb =  np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
vwkprb =  np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)

eigenwkprb0 = np.zeros(nnFlB)*complex(1.0,0.0)
uwkprb0 =  np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
vwkprb0 =  np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
invlambdamat = np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
iomrange = 1000
ompfrq0 = np.zeros(2*iomrange+1)
Kabs0 = np.zeros(2*iomrange+1)
imchi = np.zeros(2*iomrange+1)
Floquet_weakprobe_eigen(omega_p)
eigenwkprb0 = eigenwkprb
print(eigenwkprb[1])
uwkprb0 = uwkprb
vwkprb0 = vwkprb

for iom in range(0,2*iomrange+1):
    ompfrq0[iom] = 1.0*(iom-iomrange)/250.
    omega_p_var = complex(twopi,0.0)*(ompfrq0[iom]+100.0)

    for k in range(0,nnFlB):
        invlambdamat[k,k] = 1.0/(eigenwkprb0[k] + omega_p_var)

    xx = vwkprb0.dot(invlambdamat.dot(uwkprb0.dot(bvec)))

    chi = dipmom_1*xx[indFl10[0]] + dipmom_2*xx[indFl20[0]]

    #print(chi)
      
    Kabs0[iom] = np.sqrt(1.0+chi).imag
    if iom == 0:
        print(iom,omega_p,Kabs0[iom])
    
#######################################################    
#  Part (b) of the figure, with the coupling field.


btestf2 = 1.0/10.0  # Reduce E_c by a factor of 10 compared to the other figures

OmegaB = np.zeros((ndim,ndim))*complex(1.0,0.0)
OmegaB[1,3] = Omega_13*btestf2
OmegaB[1,4] = Omega_14*btestf2
OmegaB[2,3] = Omega_23*btestf2
OmegaB[2,4] = Omega_24*btestf2
for j in range(ndimA, ndim-1):
    for i in range(j+1, ndim):
        OmegaB[i,j] = np.conj(OmegaB[j,i])


eigenwkprb1 = np.zeros(nnFlB)*complex(1.0,0.0)
uwkprb1 =  np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
vwkprb1 =  np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
invlambdamat = np.zeros((nnFlB,nnFlB))*complex(1.0,0.0)
iomrange = 1000
ompfrq1 = np.zeros(2*iomrange+1)
Kabs1 = np.zeros(2*iomrange+1)
imchi = np.zeros(2*iomrange+1)
Floquet_weakprobe_eigen(omega_p)
eigenwkprb1 = eigenwkprb
print(eigenwkprb[1])
uwkprb1 = uwkprb
vwkprb1 = vwkprb
for iom in range(0,2*iomrange+1):
    ompfrq1[iom] = 1.0*(iom-iomrange)/250.0
    omega_p_var = complex(twopi,0.0)*(ompfrq1[iom]+100.0)

    for k in range(0,nnFlB):
        invlambdamat[k,k] = 1.0/(eigenwkprb1[k] + omega_p_var)

    xx = vwkprb1.dot(invlambdamat.dot(uwkprb1.dot(bvec)))

    chi = dipmom_1*xx[indFl10[0]] + dipmom_2*xx[indFl20[0]]

    Kabs1[iom] = np.sqrt(1.0+chi).imag
    if iom == 0:
        print(iom,omega_p,Kabs1[iom])


####################################################
#  Plot the results (and generate the "error bars")
        
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'


import matplotlib.gridspec as gridspec

fig = plt.figure(tight_layout=True, figsize=(9.4,3.5*0.88*1.01))
gs = gridspec.GridSpec(1, 2, width_ratios=[1,2.1])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

ax0.set_ylim([0.0007,1.6])
ax0.plot(alpha,abs(imr01wkprb_0), color='k', linestyle='dashdot', lw = 0.8)
ax0.plot(alpha,abs(imr02wkprb_0), color='k', linestyle='dashdot', lw = 0.8)
ax0.plot(alpha,abs(r00_0), color='#1f77b4', linestyle='solid')
ax0.plot(alpha,abs(r00_2), color='#1f77b4', linestyle=(0,(10,3)))
ax0.plot(alpha,abs(imr01_0), color='#ff7f0e', linestyle='solid')
ax0.plot(alpha,abs(imr02_0), color='#2ca02c', linestyle='solid')
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$\alpha$', fontsize=15)
ax0.tick_params(direction='in',bottom=True,top=False,left=True,right=False)
ax0.tick_params(which='minor',direction='in',bottom=True,top=False,left=True,right=False)
ax0.text(20.0,2.0,'${(a)}$',fontsize =15)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

factor = 1e5  # Used to give K convenient values in arbitrary units

ax1.plot(ompfrq0,Kabs0*factor,color='r',linestyle='dotted')
ax1.plot(ompfrq1,Kabs1*factor,color='r')
ax1.set_ylim([-8.0,60.0])


### Generate the plot the "error bars". The way the calculation is 
### organised is a bit messy and inefficient, but it works.
Kabskappa = np.zeros(2*iomrange+1)
nbars = 1
imchimaxmax = 0
nquasienergies = 0
for kappa in range(0,nnFlB):
    for iom in range(0,2*iomrange+1):
        #ompfrq1[iom] = 1.0*(iom-iomrange)/250 + 2.5
        omega_p_kappa = complex(twopi,0.0)*(ompfrq1[iom]+100.0)
        for k in range(0,nnFlB):
            invlambdamat[k,k] = complex(0.0,0.0)
            if k == kappa:
                invlambdamat[k,k] = 1.0/(eigenwkprb1[k] + omega_p_kappa)
        xx = vwkprb1.dot(invlambdamat.dot(uwkprb1.dot(bvec)))
        chi = dipmom_1*xx[indFl10[0]] + dipmom_2*xx[indFl20[0]]
        imchi[iom] = abs(chi.imag)
        Kabskappa[iom] = np.sqrt(1.0+chi).imag
    maxkappa = np.amax(np.abs(Kabskappa))
    imchimax = np.amax(imchi)
    if imchimax > imchimaxmax:
        imchimaxmax = imchimax
    if imchimax >= 0.000126*0.01:
        print(kappa,nnFlB,-eigenwkprb[kappa]/twopi)
        xcentre = -eigenwkprb1[kappa].real/twopi - 100.0
        xlength =  eigenwkprb1[kappa].imag/twopi
        ycentre = nbars*4.0e-6-0.4e-6
        if ycentre*factor > 2.5: #3.3:
            if xlength > 0.0: # 0.3:
               ycentre = ycentre - 2.4/factor
        if xcentre < -3.0:
            if xlength > 0.0: # 0.3:
               ycentre = ycentre - 2.4/factor
        if xcentre+xlength >= -4 and xcentre-xlength <= 4:
            ax1.errorbar(xcentre,ycentre*factor*5-7.0,xerr=xlength,color='k',fmt='+')
            nbars = nbars+1
            nquasienergies = nquasienergies + 1
    else:
        print(kappa,nnFlB)
print('Number of quasienergies in the plot:',nquasienergies)

#  Add the "error bars" for the no-coupling-field case:
ax1.errorbar(-0.4,9.0,xerr=3.6/2.0,color='grey',fmt='+')
ax1.errorbar(-1.0,12.0,xerr=0.9/2.0,color='grey',fmt='+')

ax1.set_xlabel(r'$\Delta \omega_\mathrm{p}$ $\mathrm{(}2\pi$ $\mathrm{GHz)}$', fontsize=15)
ax1.set_ylabel(r'$K$ $\mathrm{(}\mathrm{arb. un.)}$', fontsize=15)
ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(MultipleLocator(10.0))
ax1.tick_params(direction='in',bottom=True,top=False,left=True,right=False)
ax1.tick_params(which='minor',direction='in',bottom=True,top=False,left=True,right=False)
ax1.text(-0.25,61.6,'${(b)}$',fontsize =15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

#plt.savefig('figure2.pdf', format='pdf')
plt.show()                  
                    