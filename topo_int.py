# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:56:04 2021

final project - Topology in Condensed Matter - SoSe 2021

References:
    Bauer, Dieter; Hansen, Kenneth K. (2018): High-Harmonic Generation in Solids with and without Topological Edge States. In: Physical review letters 120 (17), S. 177401. DOI: 10.1103/PhysRevLett.120.177401.

@author: Max
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import scipy.integrate
import scipy.fft
import time
import concurrent.futures


tic = time.time()

A_0         =   0.1
omega       =   0.0075
n_cyc       =   5
N           =   50


t0 = 0
tf = 4000/3 *np.pi
dt = 0.001

w = -np.exp(-1.7)
v = -np.exp(-2.3)
N_cells = 200
t_eval = np.arange(t0,tf,dt)
k_points = np.linspace(-np.pi,np.pi,N_cells)


"""
H = np.array([[0,v*np.exp(1j*(k+A(t))/2)+w*np.exp(-1j*(k+A(t))/2)],[v*np.exp(-1j*(k+A(t))/2)+w*np.exp(1j*(k+A(t))/2),0]])
H'= 1j/2*np.array([[0,v*np.exp(1j*(k+A(t))/2)-w*np.exp(-1j*(k+A(t))/2)],[-v*np.exp(-1j*(k+A(t))/2)+w*np.exp(1j*(k+A(t))/2),0]])
"""


def c_dot(t,c,k,A_0=0.1,omega=0.0075,n_cyc=5):
    #t: time, k: recproc momentum, c:coefficient vector
    res = np.empty(2,dtype=complex)
    res[0]=-1j*(v*np.exp(1j*(k+A(t))/2)+w*np.exp(-1j*(k+A(t))/2))*c[1]
    res[1]=-1j*(v*np.exp(-1j*(k+A(t))/2)+w*np.exp(1j*(k+A(t))/2))*c[0]
    return res#-1j*np.array([[0,v*np.exp(1j*(k+A(t))/2)+w*np.exp(-1j*(k+A(t))/2)],[v*np.exp(-1j*(k+A(t))/2)+w*np.exp(1j*(k+A(t))/2),0]])*c
@njit
def A(t,A_0=0.1,omega=0.0075,n_cyc=5):
    res = A_0*np.sin(omega*t/(2*n_cyc))**2*np.sin(omega*t)
    return res

#@njit
def ksum(args):
    (C,kpoints,t)=args
    res = 0
    for i,k in enumerate(k_points):
        #print(C[i,:],np.shape(C[i,:]))
        #print(np.conjugate(C[i,:])@C[i,:]-np.vdot(C[i,:],C[i,:]))
        C[i,:]=np.array([C[i,:]])
        #print(C[i,:],np.shape(C[i,:]))
        H_dk = 1j/2*np.array([[0,v*np.exp(1j*(k+A(t))/2)-w*np.exp(-1j*(k+A(t))/2)],[-v*np.exp(-1j*(k+A(t))/2)+w*np.exp(1j*(k+A(t))/2),0]])
        #print(np.shape(H_k))
        res += np.conjugate(C[i,:]) @ H_dk @ C[i,:]
    return res


times = []
coeffs= []




def ode_t(k):
    c0 = 1/np.sqrt(2) * np.array([1,-((w+v)*np.cos(k/2)+1j*(w-v)*np.sin(k/2))/np.sqrt(((w+v)*np.cos(k/2))**2+((w-v)*np.sin(k/2))**2)])
    sol = scipy.integrate.solve_ivp(c_dot,t_span=[t0,tf],y0=c0,args=(k,),t_eval=t_eval,method="RK45")
    return sol
with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
    results=executor.map(ode_t,k_points)

for result in results:
    times.append(result.t)
    coeffs.append(result.y)



times = np.array(times)
coeffs= np.array(coeffs)

vel_t = []

#for ti in range(0,np.size(t_eval)):
#    vel_t.append(1/N_cells*ksum(coeffs[:,:,ti],k_points,times[0,ti]))


k_par = []
for ti in range(0,np.size(t_eval)):
    k_par.append((coeffs[:,:,ti],k_points,times[0,ti]))


with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
    results_t=executor.map(ksum,k_par)

for result_t in results_t:
    vel_t.append(1/N_cells*result_t)

"""
plt.semilogy(t_eval,vel_t)

"""
vel_t_dot=vel_t.copy()
for i in range(np.size(vel_t)):
    if not (i == 0 or i==np.size(vel_t)-1):
        vel_t_dot[i] = (vel_t[i+1]-vel_t[i-1])/(2*dt)

vel_w = scipy.fft.fftshift(scipy.fft.fft(vel_t_dot))
freqw = scipy.fft.fftshift(scipy.fft.fftfreq(np.size(t_eval)))

zerop = int((np.size(t_eval)-1)/2)+1

plt.semilogy(freqw[zerop:]/omega,np.absolute(vel_w[zerop:])**2/freqw[zerop:])
plt.savefig("intensity.jpg")
toc = time.time()

print(f'The algorithm needed {toc-tic}s.')
