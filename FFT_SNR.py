#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:37:28 2020

@author: jorge guizar alfaro
"""
import  numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from numpy.random import randn
plt.cla()
plt.clf()
plt.close('all')
##################################
def ANoise(s,SNR):
    """
    Compute d=s+n such that SNR=PS/PS
    based on the code of Mauriccio Sacchi
    Parameters
    ----------
    s : TYPE
        DESCRIPTION.
    SNR : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    n=randn(len(s));
    Es=np.sum(s**2)
    En=np.sum(n**2)
    alpha=np.sqrt(Es/(SNR*En))
    d=s+alpha*n
    return d,alpha*n

def SNRdb2SNR(SNRdB):
    """
    TRansform sigal noise ratio in dB to ratio
    

    Parameters
    ----------
    SNRdb : TYPE db SNr
        DESCRIPTION.

    Returns
    -------
    SNR : TYPE in fraction
        DESCRIPTION.

    """
    SNR=10**(SNRdB/10)
    #SNR=(10**(SNRdB)**1/10)
    
    return SNR
    
    
##################################################################
# number of samples
N=100
# time samplig
Fs=500
dt=1./Fs
# definition of frequencies
f1=50
f2=20
f3=60
t=np.linspace(0,N*dt,N)# number of samples
# isgnal of two sinusoids with frequency 50 and 80 hz
y1=np.sin(2*np.pi*t*f1)+np.sin(2*np.pi*t*f2)+.5*np.sin(2*np.pi*t*f3)

####################################################################
yf=fft(y1)
# create the frequency by 0 to 1/2dt=250 Hz  with 50 samples N//2=50
w=np.linspace(0.0, 1.0/(2.0*dt), N//2)

###################################################
SNRdB=20# dBels
SNR=SNRdb2SNR(SNRdB)

y1NOISE,NOISE=ANoise(y1,SNR)

yfNOISE=fft(y1NOISE)
###################################################

fig, axs = plt.subplots(2,2)
#fig.suptitle('Vertically stacked subplots')
axs[0,0].plot(t,y1)
axs[0,0].set_title('Signal')
axs[1,0].plot(w,2.0/N * np.abs(yf[0:N//2]))
axs[1,0].set_title('FFT')
###############################
axs[0,1].plot(t,y1NOISE)
axs[0,1].plot(t,NOISE)
axs[0,1].set_title(r'Signal+Noise %i in $dB$' %SNRdB)
axs[1,1].plot(w,2.0/N * np.abs(yfNOISE[0:N//2]))
axs[1,1].set_title('FFT')
plt.tight_layout()







