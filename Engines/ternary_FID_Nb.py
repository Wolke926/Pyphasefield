# FID ternary @paper Zhang and Steinbach - multi
#free energy using from CALPHAD
# 尚未定好的参数： miuAb界面能
import numpy as np
import os
import pandas as pd
import sympy as sp
import symengine as se
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import math
from pathlib import Path
import time
from pyphasefield.field import Field
from pyphasefield.simulation import Simulation
from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE, make_seed
import random
try:
    from numba import cuda
    import numba
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
except:
    import pyphasefield.jit_placeholder as numba
    import pyphasefield.jit_placeholder as cuda
#reference:
#1.Phase-field model with finite interface dissipation: Extension... Zhang
#2.Diffusivities of an Al–Fe–Ni melt   zhang
#3.Kinetic modeling of diffusion mobilities in bcc Ti–Nb alloys Liu
#4. Diffusion and atomic mobility of BCC Ti-Al-Nb alloys  Gu
@numba.jit
def safe_log(x):
    return math.log(x)


@numba.jit
def f_L(X, Y, T, R, VM):
    Z  =  1 - X - Y
    gibbsliq_Ti = -19887.006 + 298.7367 * T - 46.29 * T * safe_log(T)
    gibbsliq_Nb =  21262.202 + 131.229057*T - 26.4711*T * safe_log(T)+0.203475E-3*T**2 - 0.350119E-6*T**3 + 93399/T - 3.06098E-23*T**7
    gibbsliq_Al = -795.996 + 177.430178 * T - 31.748192 * T * safe_log(T)

    liq_L_Al_Nb_0 = -80470 + 23.476 * T
    liq_L_Al_Nb_1 = -18176.2 + 14.282 * T
    liq_L_Al_Nb_2 =  11868
    liq_L_Al_Ti_0 = -108250 + 38 * T
    liq_L_Al_Ti_1 = -6000 + 5 * T
    liq_L_Al_Ti_2 = 15000
    liq_L_Nb_Ti_0 = 3000
    liq_L_Al_Nb_Ti_0 = - 49466.3 - 0.2377 * T
    liq_L_Al_Nb_Ti_1 = 509335.7 - 239.65 * T
    liq_L_Al_Nb_Ti_2 =  28183.0 - 3.303 * T

    f_id = R * T * (X * safe_log(X) + Y * safe_log(Y) + Z * safe_log(Z))+ (gibbsliq_Ti * Y + gibbsliq_Nb * Z + gibbsliq_Al * X)
    term1 = X * Z *(liq_L_Al_Nb_0 + liq_L_Al_Nb_1 * (X-Z) + liq_L_Al_Nb_2 * (X-Z)**2 )
    term2 = X * Y * (liq_L_Al_Ti_0  + liq_L_Al_Ti_1 * (X-Y) + liq_L_Al_Ti_2 * (X-Y)**2 )
    term3 = Y * Z * liq_L_Nb_Ti_0
    term4 = X * Y* Z * ( X * liq_L_Al_Nb_Ti_0 + Z * liq_L_Al_Nb_Ti_1 + Y * liq_L_Al_Nb_Ti_2 )
    return (f_id + term1 + term2 + term3 + term4)/VM


@numba.jit
def f_S(X,Y,T,R,VM):

    Z = 1-X-Y

    gibbsBCC_Ti = 26483.26 - 182.426271 * T + 19.0900905 * T * safe_log(T) - 22.00832E-3*T**2 + 1.228863E-6 * T**3 + 1400501/T
    gibbsBCC_Nb = -8519.353 + 142.045474 * T - 26.4711 * T * safe_log(T) + 0.203475E-3 * T**2 - 0.35012E-6 * T**3 + 93399/T
    gibbsBCC_Al = -1195.378 + 183.871153 * T - 31.748192 * T * safe_log(T) - 1230.622E25 * T**(-9)

    BCC_L_Al_Nb_0= -58005.3 + 13 * T
    BCC_L_Al_Nb_1= 26362.2 - 9.764 * T
    BCC_L_Al_Nb_2= 0
    BCC_L_Al_Ti_0= -128500 + 39 * T
    BCC_L_Al_Ti_1= 6000
    BCC_L_Al_Ti_2= 21200
    BCC_L_Nb_Ti_0= 8900
    BCC_L_Al_Nb_Ti_0 = -21442.5 - 13.647 * T
    BCC_L_Al_Nb_Ti_1 = -3640.44 + 1.029 * T
    BCC_L_Al_Nb_Ti_2 = -372108.8 + 212.18 * T

    f_id = R * T * (X * safe_log(X) + Y * safe_log(Y) + Z * safe_log(Z))+ (gibbsBCC_Ti * Y + gibbsBCC_Nb * Z + gibbsBCC_Al * X)
    term1 = X * Z * (BCC_L_Al_Nb_0 + BCC_L_Al_Nb_1 * (X - Z) + BCC_L_Al_Nb_2 * (X - Z) ** 2)
    term2 = X * Y * (BCC_L_Al_Ti_0 + BCC_L_Al_Ti_1 * (X - Y) + BCC_L_Al_Ti_2 * (X - Y) ** 2)  
    term3 = Y * Z * BCC_L_Nb_Ti_0
    term4 = X * Y * Z * (X * BCC_L_Al_Nb_Ti_0 + Z * BCC_L_Al_Nb_Ti_1 + Y * BCC_L_Al_Nb_Ti_2)
    return (f_id + term1 + term2 + term3 + term4)/VM
#################################################################################################
#B14
@numba.jit
def DgibbsBCC_1(X,Y,T,R,VM):
    Z = 1- X-Y
    gibbsBCC_Ti = 26483.26 - 182.426271 * T + 19.0900905 * T * safe_log(T) - 22.00832E-3*T**2 + 1.228863E-6 * T**3 + 1400501/T
    gibbsBCC_Nb = -8519.353 + 142.045474 * T - 26.4711 * T * safe_log(T) + 0.203475E-3 * T**2 - 0.35012E-6 * T**3 + 93399/T
    gibbsBCC_Al = -1195.361 + 183.871153 * T - 31.748192 * T * safe_log(T) - 1230.622E25 * T**(-9)

    BCC_L_Al_Nb_0= -58005.3 + 13 * T
    BCC_L_Al_Nb_1= 26362.2 - 9.764 * T
    BCC_L_Al_Nb_2= 0
    BCC_L_Al_Ti_0= -128500 + 39 * T
    BCC_L_Al_Ti_1= 6000
    BCC_L_Al_Ti_2= 21200
    BCC_L_Nb_Ti_0= 8900
    BCC_L_Al_Nb_Ti_0 = -21442.5 - 13.647 * T
    BCC_L_Al_Nb_Ti_1 = -3640.44 + 1.029 * T
    BCC_L_Al_Nb_Ti_2 = -372108.8 + 212.18 * T

    D_id = gibbsBCC_Al - gibbsBCC_Nb + R * T * (safe_log(X) - safe_log(Z))

    term1 = (Z - X) * (BCC_L_Al_Nb_0 + BCC_L_Al_Nb_1 * (X - Z) + BCC_L_Al_Nb_2 * (X - Z) ** 2) + (X * Z) * (2 * BCC_L_Al_Nb_1 + 4 * BCC_L_Al_Nb_2 * (X - Z))
    term2 = Y * (BCC_L_Al_Ti_0 + BCC_L_Al_Ti_1 * (X - Y) + BCC_L_Al_Ti_2 * (X - Y) ** 2) + X * Y * (
                BCC_L_Al_Ti_1 + 2 * BCC_L_Al_Ti_2 * (X - Y))

    term3 = -Y* BCC_L_Nb_Ti_0
    Dex1_DX = term1 + term2 + term3
    Dex2_DX = (Y - 2 * X * Y - Y ** 2) * (X * BCC_L_Al_Nb_Ti_0 + Z * BCC_L_Al_Nb_Ti_1 + Y * BCC_L_Al_Nb_Ti_2) + (X * Y * Z) * (BCC_L_Al_Nb_Ti_0 - BCC_L_Al_Nb_Ti_1)

    return (D_id + Dex1_DX + Dex2_DX) / VM

@numba.jit
def Dgibbsliq_1(X, Y, T, R, VM):
    Z  =  1 - X - Y
    gibbsliq_Ti = -19887.006 + 298.7367 * T - 46.29 * T * safe_log(T)
    gibbsliq_Nb =  21262.202 + 131.229057*T - 26.4711*T * safe_log(T)+0.203475E-3*T**2 - 0.350119E-6*T**3 + 93399/T - 3.06098E-23*T**7
    gibbsliq_Al = -795.991 + 177.430178 * T - 31.748192 * T * safe_log(T)

    liq_L_Al_Nb_0 = -80470 + 23.476 * T
    liq_L_Al_Nb_1 = -18176.2 + 14.282 * T
    liq_L_Al_Nb_2 =  11868
    liq_L_Al_Ti_0 = -108250 + 38 * T
    liq_L_Al_Ti_1 = -6000 + 5 * T
    liq_L_Al_Ti_2 = 15000
    liq_L_Nb_Ti_0 = 3000
    liq_L_Al_Nb_Ti_0 = - 49466.3 - 0.2377 * T
    liq_L_Al_Nb_Ti_1 = 509335.7 - 239.65 * T
    liq_L_Al_Nb_Ti_2 =  28183.0 - 3.303 * T

    D_id = gibbsliq_Al - gibbsliq_Nb + R * T * (safe_log(X) - safe_log(Z))
    term1 = (Z - X) * (liq_L_Al_Nb_0 + liq_L_Al_Nb_1 * (X - Z) + liq_L_Al_Nb_2 * (X - Z) ** 2) + (X * Z) * (2 * liq_L_Al_Nb_1 + 4 * liq_L_Al_Nb_2 * (X - Z))
    term2 = Y * (liq_L_Al_Ti_0 + liq_L_Al_Ti_1 * (X - Y) + liq_L_Al_Ti_2 * (X - Y) ** 2) + X * Y * (
                liq_L_Al_Ti_1 + 2 * liq_L_Al_Ti_2 * (X - Y))

    term3 = -Y* liq_L_Nb_Ti_0
    Dex1_DX = term1 + term2 + term3
    Dex2_DX = (Y - 2 * X * Y - Y ** 2) * (X * liq_L_Al_Nb_Ti_0 + Z * liq_L_Al_Nb_Ti_1 + Y * liq_L_Al_Nb_Ti_2) + (X * Y * Z) * (liq_L_Al_Nb_Ti_0 - liq_L_Al_Nb_Ti_1)

    return (D_id + Dex1_DX + Dex2_DX) / VM

@numba.jit
def DgibbsBCC_2(X,Y,T,R,VM):
    Z  =  1 - X - Y
    gibbsBCC_Ti = 26483.26 - 182.426271 * T + 19.0900905 * T * safe_log(T) - 22.00832E-3*T**2 + 1.228863E-6 * T**3 + 1400501/T
    gibbsBCC_Nb = -8519.353 + 142.045474 * T - 26.4711 * T * safe_log(T) + 0.203475E-3 * T**2 - 0.35012E-6 * T**3 + 93399/T
    gibbsBCC_Al = -1195.361 + 183.871153 * T - 31.748192 * T * safe_log(T) - 1230.622E25 * T**(-9)

    BCC_L_Al_Nb_0= -58005.3 + 13 * T
    BCC_L_Al_Nb_1= 26362.2 - 9.764 * T
    BCC_L_Al_Nb_2= 0
    BCC_L_Al_Ti_0= -128500 + 39 * T
    BCC_L_Al_Ti_1= 6000
    BCC_L_Al_Ti_2= 21200
    BCC_L_Nb_Ti_0= 8900
    BCC_L_Al_Nb_Ti_0 = -21442.5 - 13.647 * T
    BCC_L_Al_Nb_Ti_1 = -3640.44 + 1.029 * T
    BCC_L_Al_Nb_Ti_2 = -372108.8 + 212.18 * T

    D_id_DY = gibbsBCC_Ti - gibbsBCC_Nb + R*T *(safe_log(Y) - safe_log(Z))
    term1 = -X*(BCC_L_Al_Nb_0 + BCC_L_Al_Nb_1*(X-Z) +BCC_L_Al_Nb_2 *(X-Z)**2) + (X*Z)*(BCC_L_Al_Nb_1 + 2*BCC_L_Al_Nb_2 * (X-Z))
    term2 = X*(BCC_L_Al_Ti_0 + BCC_L_Al_Ti_1 * (X-Y) + BCC_L_Al_Ti_2*(X-Y)**2) + X*Y*(- BCC_L_Al_Ti_1 - 2*BCC_L_Al_Ti_2*(X-Y))
    term3 = (1- X - 2*Y)* BCC_L_Nb_Ti_0
    Dex1_DY = term1 + term2 + term3
    Dex2_DY = (X - 2 * X * Y - X ** 2) * (X * BCC_L_Al_Nb_Ti_0 + Z * BCC_L_Al_Nb_Ti_1 + Y * BCC_L_Al_Nb_Ti_2) + (X * Y *Z) * ( - BCC_L_Al_Nb_Ti_1 + BCC_L_Al_Nb_Ti_2)

    return (D_id_DY + Dex1_DY + Dex2_DY)/VM

@numba.jit
def Dgibbsliq_2(X, Y, T, R, VM):
    Z = 1.0 - X - Y

    gibbsliq_Ti = -19887.006 + 298.7367 * T - 46.29 * T * safe_log(T)
    gibbsliq_Nb =  21262.202 + 131.229057*T - 26.4711*T * safe_log(T)+0.203475E-3*T**2 - 0.350119E-6*T**3 + 93399/T - 3.06098E-23*T**7
    gibbsliq_Al = -795.991 + 177.430178 * T - 31.748192 * T * safe_log(T)

    liq_L_Al_Nb_0 = -80470 + 23.476 * T
    liq_L_Al_Nb_1 = -18176.2 + 14.282 * T
    liq_L_Al_Nb_2 =  11868.0
    liq_L_Al_Ti_0 = -108250 + 38 * T
    liq_L_Al_Ti_1 = -6000 + 5 * T
    liq_L_Al_Ti_2 = 15000.0
    liq_L_Nb_Ti_0 = 3000.0
    liq_L_Al_Nb_Ti_0 = - 49466.3 - 0.2377 * T
    liq_L_Al_Nb_Ti_1 = 509335.7 - 239.65 * T
    liq_L_Al_Nb_Ti_2 =  28183.0 - 3.303 * T

    D_id_DY = gibbsliq_Ti - gibbsliq_Nb + R*T *(safe_log(Y) - safe_log(Z))
    term1 = -X*(liq_L_Al_Nb_0 + liq_L_Al_Nb_1*(X-Z) +liq_L_Al_Nb_2 *(X-Z)**2) + (X*Z)*(liq_L_Al_Nb_1 + 2*liq_L_Al_Nb_2 * (X-Z))
    term2 = X*(liq_L_Al_Ti_0 + liq_L_Al_Ti_1 * (X-Y) + liq_L_Al_Ti_2*(X-Y)**2) + X*Y*(- liq_L_Al_Ti_1 - 2*liq_L_Al_Ti_2*(X-Y))

    term3 = (1- X - 2*Y)* liq_L_Nb_Ti_0
    Dex1_DY = term1 + term2 + term3
    Dex2_DY = (X - 2 * X * Y - X ** 2) * (X * liq_L_Al_Nb_Ti_0 + Z * liq_L_Al_Nb_Ti_1 + Y * liq_L_Al_Nb_Ti_2) + (X * Y *Z) * (- liq_L_Al_Nb_Ti_1 + liq_L_Al_Nb_Ti_2)

    return (D_id_DY + Dex1_DY + Dex2_DY)/VM

@cuda.jit
def solvePhi(fields,params,fields_out):
    phi = fields[0]
    c1 = fields[1]
    c1_s = fields[2]
    c1_l = fields[3]

    c2 = fields[4]
    c2_s = fields[5]
    c2_l = fields[6]
    Tarr = fields[7]
    beta = fields[16]

    phi_out = fields_out[0]
    c1_out = fields_out[1]
    c2_out = fields_out[4]

    dx = params[0]
    dt = params[1]
    P1 = params[2]
    P2 = params[3]
    mu = params[4] # ?? const
    Vs = params[5]
    G = params[6]
    dT = params[7]
    initXpos = params[8]
    sigma = params[9]
    R = params[10]
    VM = params[11]
    me_1 = params[16]
    me_2 = params[17]
    k_an = params[18]
    v_an = params[19]
    xi = params[23]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1,phi.shape[0] - 1,stridey):
        for j in range(startx + 1,phi.shape[1] -1,stridex):

            T = Tarr[i][j]
            f_s = f_S(c1_s[i][j], c2_s[i][j], T, R, VM)
            f_l = f_L(c1_l[i][j], c2_l[i][j], T, R, VM)

            mu1_s = DgibbsBCC_1(c1_s[i][j], c2_s[i][j], T, R, VM)
            mu1_l = Dgibbsliq_1(c1_l[i][j], c2_l[i][j], T, R, VM)

            mu2_s = DgibbsBCC_2(c1_s[i][j], c2_s[i][j], T, R, VM)
            mu2_l = Dgibbsliq_2(c1_l[i][j],c2_l[i][j], T, R, VM)


            dphidy = (phi[i][j+1]-phi[i][j-1])/(2*dx)
            dphidx = (phi[i+1][j]-phi[i-1][j])/(2*dx)
            normdenom = math.sqrt(dphidx**2 + dphidy**2)

            if normdenom > 1e-9:
                normx = dphidx/normdenom
                normy = dphidy/normdenom
            else:
                normx = 0.0
                normy = 0.0

            sigma_n = sigma * (1 - v_an * (3 - 4 * (normx**4 + normy**4)))
            mu_n = mu * (1 - v_an * (3 - 4 * (normx**4 + normy**4))) # miu_alpha_beta: interfacial mobility
           
            #Calculate K (kinetic coefficient) and dG (driving force)

            #(37) mu_n interface mob (aniso)
            SUM_ = (c1_l[i][j] - c1_s[i][j])**2/P1 + (c2_l[i][j] - c2_s[i][j])**2/P2
            K = 8 * xi * mu_n / (8 * xi + mu_n * math.pi**2 * SUM_)

            #(38)
            phi_l = 1 - phi[i][j]
            sum_1 = (phi[i][j]*mu1_s + phi_l*mu1_l)*(c1_l[i][j]-c1_s[i][j])
            sum_2 = (phi[i][j]*mu2_s + phi_l*mu2_l)*(c2_l[i][j]-c2_s[i][j])
            dG = f_l - f_s - sum_1 - sum_2

            laplacian_phi = (0.5 * (phi[i][j-1] + phi[i][j+1] + phi[i+1][j] + phi[i-1][j] + \
                                                0.5 * (phi[i+1][j+1] + phi[i-1][j+1] + phi[i-1][j-1] + \
                                                    phi[i+1][j-1]) - 6 * phi[i][j])) / (dx * dx)

            laplacian_phi_l = (0.5 * ((1-phi[i][j-1]) + (1-phi[i][j+1]) + (1-phi[i+1][j]) + (1-phi[i-1][j]) + \
                                                0.5 * (4 - phi[i+1][j+1] - phi[i-1][j+1] - phi[i-1][j-1] - \
                                                    phi[i+1][j-1]) - 6 * (1- phi[i][j]))) / (dx * dx)

            # (33) I_s = I_alpha, I_l = I_beta [1]
            I_s = laplacian_phi + (math.pi**2/(xi**2)) * phi[i][j]
            I_l = laplacian_phi_l + (math.pi**2/(xi**2)) * (1 - phi[i][j])
            #39 [1]
            dphidt =  K * (sigma_n * (laplacian_phi + math.pi**2 / xi**2 * (phi[i][j] - 0.5)) + math.pi/xi * math.sqrt(phi[i][j] * (1-phi[i][j])) * dG)
            curr_phi = phi[i][j]

            eta = 0.01
            betaval = beta[i][j] - 0.5
          
            # phi_out[i,j] = curr_phi + dt*dphidt + eta*betaval*math.sqrt(dt)
            phi_out[i, j] = curr_phi + dt * dphidt

            if phi_out[i][j] < 0.0:
                phi_out[i][j] = 0
            elif phi_out[i][j] > 1.0:
                phi_out[i][j] = 1

#boundary condition##### 注意仅当i+1 等被使用时候才需要进行这一步
    for i in range(starty, phi.shape[0], stridey):
        for j in range(startx, phi.shape[1], stridex):
            if i == 0 or i == phi.shape[0]:
               for j in range(phi.shape[1]):
                  phi_out[0][j] = phi_out[1][j]
                  phi_out[phi.shape[0]][j] = phi_out[phi.shape[0] - 1][j]

            if j == 0 or j == phi.shape[1]:
               for i in range(phi.shape[0]):
                   phi_out[i][0] = phi_out[i][1]
                   phi_out[i][phi.shape[1]] = phi_out[i][phi.shape[1] - 1]


@cuda.jit
def solveC_Dij(fields,params,fields_out):

    phi = fields[0]
    c1 = fields[1]
    c1_s = fields[2]
    c1_l = fields[3]

    c2 = fields[4]
    c2_s = fields[5]
    c2_l = fields[6]
    Tarr = fields[7]

    D_11_s = fields[8]
    D_12_s = fields[9]
    D_21_s = fields[10]
    D_22_s = fields[11]
    D_11_l = fields[12]
    D_12_l = fields[13]
    D_21_l = fields[14]
    D_22_l = fields[15]

    phi_out = fields_out[0]
    c1_out = fields_out[1]
    c1_s_out = fields_out[2]
    c1_l_out = fields_out[3]

    c2_out = fields_out[4]
    c2_s_out = fields_out[5]
    c2_l_out = fields_out[6]

    dx = params[0]
    dt = params[1]
    P1 = params[2]
    P2 = params[3]
    mu = params[4]  # ?? const

    initXpos = params[8]
    sigma = params[9]
    R = params[10]
    VM = params[11]
    ke_1 = params[12]  # kini
    ke_2 = params[13]
    c1_0 = params[14]
    c2_0 = params[15]
    me_1 = params[16]
    me_2 = params[17]
    k_an = params[18]
    v_an = params[19]
    Tn = params[20]
    Ts = params[21]

    xi = params[23]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):

            T = Tarr[i][j]

            #phi next has been calculated in solv_phi
            dphidt_ij = (phi_out[i][j] - phi[i][j]) / dt
            dphidt_opp = ((1-phi_out[i,j]) - (1-phi[i,j])) / dt

            mu1_s = DgibbsBCC_1(c1_s[i][j], c2_s[i][j], T, R, VM)  #
            mu1_l = Dgibbsliq_1(c1_l[i][j], c2_l[i][j], T, R, VM)  #
            mu2_s = DgibbsBCC_2(c1_s[i][j], c2_s[i][j], T, R, VM)  #
            mu2_l = Dgibbsliq_2(c1_l[i][j], c2_l[i][j], T, R, VM)  #

#***************************   c1 calc ***********************************#

            # i = 1, j =1,2, alpha = s
            x1plus_s = ((phi_out[i][j] + phi_out[i+1][j]) / 2) * (D_11_s[i][j] *(c1_s[i+1][j] - c1_s[i][j]) / dx + D_12_s[i][j] * (c2_s[i+1][j] - c2_s[i][j]) / dx)
            x1mins_s = ((phi_out[i][j] + phi_out[i-1][j]) / 2) * (D_11_s[i][j] *(c1_s[i-1][j] - c1_s[i][j]) / dx+ D_12_s[i][j] * (c2_s[i-1][j] - c2_s[i][j]) / dx)
            y1plus_s = ((phi_out[i][j] + phi_out[i][j+1]) / 2) * (D_11_s[i][j] * (c1_s[i][j+1] - c1_s[i][j])/ dx + D_12_s[i][j] * (c2_s[i][j+1] - c2_s[i][j]) / dx)
            y1mins_s = ((phi_out[i][j] + phi_out[i][j-1]) / 2) * (D_11_s[i][j] * (c1_s[i][j-1] - c1_s[i][j])/ dx + D_12_s[i][j] * (c2_s[i][j-1] - c2_s[i][j]) / dx)


            #liquid component of diffusion # i = 1, j =1,2, alpha = l
            x1plus_l = (((1-phi_out[i][j])  + (1-phi_out[i+1][j]) ) / 2) * (D_11_l[i][j] * (c1_l[i+1][j] - c1_l[i][j]) / dx +  D_12_l[i][j] *(c2_l[i+1][j] - c2_l[i][j]) / dx)
            x1mins_l = (((1-phi_out[i][j])  + (1-phi_out[i-1][j]) ) / 2) * (D_11_l[i][j] * (c1_l[i-1][j] - c1_l[i][j]) / dx +  D_12_l[i][j] *(c2_l[i-1][j] - c2_l[i][j]) / dx)
            y1plus_l = (((1-phi_out[i][j])  + (1-phi_out[i][j+1]) ) / 2) * (D_11_l[i][j] * (c1_l[i][j+1] - c1_l[i][j]) / dx +  D_12_l[i][j] *(c2_l[i][j+1] - c2_l[i][j]) / dx)
            y1mins_l = (((1-phi_out[i][j])  + (1-phi_out[i][j-1]) ) / 2) * (D_11_l[i][j] * (c1_l[i][j-1] - c1_l[i][j]) / dx +  D_12_l[i][j] *(c2_l[i][j-1] - c2_l[i][j]) / dx)

            # part 1 in (17) i = 1, alpha = s
            part1_s = (x1plus_s + x1mins_s + y1plus_s + y1mins_s) / dx
            part1_l = (x1plus_l + x1mins_l + y1plus_l + y1mins_l) / dx

            # part 2 in (17) i = 1, alpha = s
            part2_s = P1 * phi_out[i][j] * (1 - phi_out[i][j]) * (mu1_l - mu1_s)
            part2_l = P1 * phi_out[i][j] * (1 - phi_out[i][j]) * (mu1_s - mu1_l)

            # part 3 in (17)
            part3_s = phi_out[i][j] * dphidt_opp * (c1_l[i][j] - c1_s[i][j])
            part3_l = (1 - phi_out[i][j]) * dphidt_ij * (c1_s[i][j] - c1_l[i][j])

            #dc/dt completed term
            if phi_out[i][j] < 1e-3: #if we are in liquid portion
                dc1sdt = 0.0
                dc1ldt = part1_l # (xplus_l + xmins_l + yplus_l + ymins_l) / dx
            elif phi_out[i][j] > 1-1e-3: #if we are in solid portion
                dc1sdt = part1_s # (xplus_s + xmins_s + yplus_s + ymins_s) / dx
                dc1ldt = 0.0
            else: #if(phi_out[i,j] >= 1e-9 && phi_out[i,j] <= 1-1e-9) #if we are at the interface solve both equations
                dc1sdt = (part1_s + part2_s - part3_s) / phi_out[i][j]
                dc1ldt = (part1_l + part2_l - part3_l) / (1-phi_out[i][j])

            c1_s_out[i][j] = c1_s[i][j] + dt * dc1sdt
            c1_l_out[i][j] = c1_l[i][j] + dt * dc1ldt

            if c1_s_out[i][j] < 0.0:
                c1_s_out[i][j] = 0.0000001
            elif c1_s_out[i][j] > 1.0:
                c1_s_out[i][j] = 0.9999999

            if c1_l_out[i][j] < 0.0:
                c1_l_out[i][j] = 0.0000001
            elif c1_l_out[i][j] > 1.0:
                c1_l_out[i][j] = 0.9999999

            c1_out[i][j] = c1_s_out[i][j] * phi_out[i][j] + c1_l_out[i][j] * (1-phi_out[i][j])

#************************ calc the c2 ***********************************************************
            #i = 2, j =1,2, alpha = s

            x2plus_s = ((phi_out[i][j] + phi_out[i + 1][j]) / 2) * ((D_21_s[i][j] * (c1_s[i + 1][j] - c1_s[i][j]) / dx) + D_22_s[i][j] * (c2_s[i + 1][j] - c2_s[i][j]) / dx)
            x2mins_s = ((phi_out[i][j] + phi_out[i - 1][j]) / 2) * ((D_21_s[i][j] * (c1_s[i - 1][j] - c1_s[i][j]) / dx) + D_22_s[i][j] * (c2_s[i - 1][j] - c2_s[i][j]) / dx)
            y2plus_s = ((phi_out[i][j] + phi_out[i][j + 1]) / 2) * ((D_21_s[i][j] * (c1_s[i][j + 1] - c1_s[i][j]) / dx) + D_22_s[i][j] * (c2_s[i][j + 1] - c2_s[i][j]) / dx)
            y2mins_s = ((phi_out[i][j] + phi_out[i][j - 1]) / 2) * ((D_21_s[i][j] * (c1_s[i][j - 1] - c1_s[i][j]) / dx) + D_22_s[i][j] * (c2_s[i][j - 1] - c2_s[i][j]) / dx)

            # liquid component of diffusion
            # i = 2, j =1,2, alpha = l , almost same as above ,difference :1.use (1-phi) instead of phi ; 2. l instead of s
            x2plus_l = (((1 - phi_out[i][j]) + (1 - phi_out[i + 1][j])) / 2) * (D_21_l[i][j] * (c1_l[i + 1][j] - c1_l[i][j]) / dx + D_22_l[i][j] * (c2_l[i + 1][j] - c2_l[i][j]) / dx)
            x2mins_l = (((1 - phi_out[i][j]) + (1 - phi_out[i - 1][j])) / 2) * (D_21_l[i][j] * (c1_l[i - 1][j] - c1_l[i][j]) / dx + D_22_l[i][j] * (c2_l[i - 1][j] - c2_l[i][j]) / dx)
            y2plus_l = (((1 - phi_out[i][j]) + (1 - phi_out[i][j + 1])) / 2) * (D_21_l[i][j] * (c1_l[i][j + 1] - c1_l[i][j]) / dx + D_22_l[i][j] * (c2_l[i][j + 1] - c2_l[i][j]) / dx)
            y2mins_l = (((1 - phi_out[i][j]) + (1 - phi_out[i][j - 1])) / 2) * (D_21_l[i][j] * (c1_l[i][j - 1] - c1_l[i][j]) / dx + D_22_l[i][j] * (c2_l[i][j - 1] - c2_l[i][j]) / dx)

            # part 1 in (17)
            part1_s = (x2plus_s + x2mins_s + y2plus_s + y2mins_s) / dx
            part1_l = (x2plus_l + x2mins_l + y2plus_l + y2mins_l) / dx

            # part 2 in (17)
            part2_s = P2 * phi_out[i][j] * (1 - phi_out[i][j]) * (mu2_l - mu2_s)
            part2_l = P2 * phi_out[i][j] * (1 - phi_out[i][j]) * (mu2_s - mu2_l)

            # part 3 in (17)
            part3_s = phi_out[i][j] * dphidt_opp * (c2_l[i][j] - c2_s[i][j])
            part3_l = (1 - phi_out[i][j]) * dphidt_ij * (c2_s[i][j] - c2_l[i][j])

            #dc/dt completed term
            if phi_out[i][j] < 1e-3: #if we are in liquid portion
                dc2sdt = 0.0
                dc2ldt = part1_l # (xplus_l + xmins_l + yplus_l + ymins_l) / dx
            elif phi_out[i][j] > 1-1e-3: #if we are in solid portion
                dc2sdt = part1_s # (xplus_s + xmins_s + yplus_s + ymins_s) / dx
                dc2ldt = 0.0
            else: #if(phi_out[i,j] >= 1e-9 && phi_out[i,j] <= 1-1e-9) #if we are at the interface solve both equations
                dc2sdt = (part1_s + part2_s - part3_s) / phi_out[i][j]
                dc2ldt = (part1_l + part2_l - part3_l) / (1-phi_out[i][j])
            # c2
            c2_s_out[i][j] = c2_s[i][j] + dt * dc2sdt
            c2_l_out[i][j] = c2_l[i][j] + dt * dc2ldt

            if c2_s_out[i][j] < 0.0:
                c2_s_out[i][j] = 0.0000001
            elif c2_s_out[i][j] > 1.0:
                c2_s_out[i][j] = 0.9999999

            if c2_l_out[i][j] < 0.0:
                c2_l_out[i][j] = 0.0000001
            elif c2_l_out[i][j] > 1.0:
                c2_l_out[i][j] = 0.9999999
            c2_out[i][j] = c2_s_out[i][j] * phi_out[i][j] + c2_l_out[i][j] * (1 - phi_out[i][j])
@cuda.jit
def solveC_const_D(fields,params,fields_out):
    phi = fields[0]
    c1 = fields[1]
    c1_s = fields[2]
    c1_l = fields[3]

    c2 = fields[4]
    c2_s = fields[5]
    c2_l = fields[6]
    Tarr = fields[7]

    phi_out = fields_out[0]
    c1_out = fields_out[1]
    c1_s_out = fields_out[2]
    c1_l_out = fields_out[3]

    c2_out = fields_out[4]
    c2_s_out = fields_out[5]
    c2_l_out = fields_out[6]

    dx = params[0]
    dt = params[1]
    P1 = params[2]
    P2 = params[3]
    mu = params[4]  # ?? const

    initXpos = params[8]
    sigma = params[9]
    R = params[10]
    VM = params[11]
    ke_1 = params[12]  # kini
    ke_2 = params[13]
    c1_0 = params[14]
    c2_0 = params[15]
    me_1 = params[16]
    me_2 = params[17]
    k_an = params[18]
    v_an = params[19]
    Tn = params[20]
    Ts = params[21]

    xi = params[23]

    Dl_11 = params[24]
    Dl_12 = params[25]
    Dl_22 = params[26]

    Ds_11 = params[27]
    Ds_12 = params[28]
    Ds_22 = params[29]
    # the D ??
    Dl_21 = params[30]
    Ds_21 = params[31]


    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):

            T = Tarr[i][j]

            #phi next has been calculated in solv_phi
            dphidt_ij = (phi_out[i][j] - phi[i][j]) / dt
            dphidt_opp = ((1-phi_out[i,j]) - (1-phi[i,j])) / dt

            mu1_s = DgibbsBCC_1(c1_s[i][j], c2_s[i][j], T, R, VM)  #
            mu1_l = Dgibbsliq_1(c1_l[i][j], c2_l[i][j], T, R, VM)  #
            mu2_s = DgibbsBCC_2(c1_s[i][j], c2_s[i][j], T, R, VM)  #
            mu2_l = Dgibbsliq_2(c1_l[i][j], c2_l[i][j], T, R, VM)  #


#***************************   c1 calc ***********************************#

            # i = 1, j =1,2, alpha = s
            x1plus_s = ((phi_out[i][j] + phi_out[i+1][j]) / 2) * ((Ds_11 *(c1_s[i+1][j] - c1_s[i][j]) / dx) + Ds_12 * (c2_s[i+1][j] - c2_s[i][j]) / dx)
            x1mins_s = ((phi_out[i][j] + phi_out[i-1][j]) / 2) * ((Ds_11 *(c1_s[i-1][j] - c1_s[i][j]) / dx) + Ds_12 * (c2_s[i-1][j] - c2_s[i][j]) / dx)
            y1plus_s = ((phi_out[i][j] + phi_out[i][j+1]) / 2) * ((Ds_11 * (c1_s[i][j+1] - c1_s[i][j])/ dx) + Ds_12 * (c2_s[i][j+1] - c2_s[i][j]) / dx)
            y1mins_s = ((phi_out[i][j] + phi_out[i][j-1]) / 2) * ((Ds_11 * (c1_s[i][j-1] - c1_s[i][j])/ dx) + Ds_12 * (c2_s[i][j-1] - c2_s[i][j]) / dx)


            #liquid component of diffusion # i = 1, j =1,2, alpha = l
            x1plus_l = (((1-phi_out[i][j])  + (1-phi_out[i+1][j]) ) / 2) * (Dl_11 * (c1_l[i+1][j] - c1_l[i][j]) / dx +  Dl_12 *(c2_l[i+1][j] - c2_l[i][j]) / dx)
            x1mins_l = (((1-phi_out[i][j])  + (1-phi_out[i-1][j]) ) / 2) * (Dl_11 * (c1_l[i-1][j] - c1_l[i][j]) / dx +  Dl_12 *(c2_l[i-1][j] - c2_l[i][j]) / dx)
            y1plus_l = (((1-phi_out[i][j])  + (1-phi_out[i][j+1]) ) / 2) * (Dl_11 * (c1_l[i][j+1] - c1_l[i][j]) / dx +  Dl_12 *(c2_l[i][j+1] - c2_l[i][j]) / dx)
            y1mins_l = (((1-phi_out[i][j])  + (1-phi_out[i][j-1]) ) / 2) * (Dl_11 * (c1_l[i][j-1] - c1_l[i][j]) / dx +  Dl_12 *(c2_l[i][j-1] - c2_l[i][j]) / dx)

            # part 1 in (17) i = 1, alpha = s
            part1_s = (x1plus_s + x1mins_s + y1plus_s + y1mins_s) / dx
            part1_l = (x1plus_l + x1mins_l + y1plus_l + y1mins_l) / dx

            # part 2 in (17) i = 1, alpha = s
            part2_s = P1 * phi_out[i][j] * (1 - phi_out[i][j]) * (mu1_l - mu1_s)
            part2_l = P1 * phi_out[i][j] * (1 - phi_out[i][j]) * (mu1_s - mu1_l)

            # part 3 in (17)
            part3_s = phi_out[i][j] * dphidt_opp * (c1_l[i][j] - c1_s[i][j])
            part3_l = (1 - phi_out[i][j]) * dphidt_ij * (c1_s[i][j] - c1_l[i][j])

            #dc/dt completed term
            if phi_out[i][j] < 1e-3: #if we are in liquid portion
                dc1sdt = 0.0
                dc1ldt = part1_l # (xplus_l + xmins_l + yplus_l + ymins_l) / dx
            elif phi_out[i][j] > 1-1e-3: #if we are in solid portion
                dc1sdt = part1_s # (xplus_s + xmins_s + yplus_s + ymins_s) / dx
                dc1ldt = 0.0
            else: #if(phi_out[i,j] >= 1e-9 && phi_out[i,j] <= 1-1e-9) #if we are at the interface solve both equations
                dc1sdt = (part1_s + part2_s - part3_s) / phi_out[i][j]
                dc1ldt = (part1_l + part2_l - part3_l) / (1-phi_out[i][j])

            c1_s_out[i][j] = c1_s[i][j] + dt * dc1sdt
            c1_l_out[i][j] = c1_l[i][j] + dt * dc1ldt

            if c1_s_out[i][j] < 0.0:
                c1_s_out[i][j] = 0.0000001
            elif c1_s_out[i][j] > 1.0:
                c1_s_out[i][j] = 0.9999999
            
            if c1_l_out[i][j] < 0.0:
                c1_l_out[i][j] = 0.0000001
            elif c1_l_out[i][j] > 1.0:
                c1_l_out[i][j] = 0.9999999

            c1_out[i][j] = c1_s_out[i][j] * phi_out[i][j] + c1_l_out[i][j] * (1-phi_out[i][j])

#************************ calc the c2 ***********************************************************
            #i = 2, j =1,2, alpha = s

            x2plus_s = ((phi_out[i][j] + phi_out[i + 1][j]) / 2) * ((Ds_21 * (c1_s[i + 1][j] - c1_s[i][j]) / dx) + Ds_22 * (c2_s[i + 1][j] - c2_s[i][j]) / dx)
            x2mins_s = ((phi_out[i][j] + phi_out[i - 1][j]) / 2) * ((Ds_21 * (c1_s[i - 1][j] - c1_s[i][j]) / dx) + Ds_22 * (c2_s[i - 1][j] - c2_s[i][j]) / dx)
            y2plus_s = ((phi_out[i][j] + phi_out[i][j + 1]) / 2) * ((Ds_21 * (c1_s[i][j + 1] - c1_s[i][j]) / dx) + Ds_22 * (c2_s[i][j + 1] - c2_s[i][j]) / dx)
            y2mins_s = ((phi_out[i][j] + phi_out[i][j - 1]) / 2) * ((Ds_21 * (c1_s[i][j - 1] - c1_s[i][j]) / dx) + Ds_22 * (c2_s[i][j - 1] - c2_s[i][j]) / dx)

            # liquid component of diffusion
            # i = 2, j =1,2, alpha = l , almost same as above ,difference :1.use (1-phi) instead of phi ; 2. l instead of s
            x2plus_l = (((1 - phi_out[i][j]) + (1 - phi_out[i + 1][j])) / 2) * (Dl_21 * (c1_l[i + 1][j] - c1_l[i][j]) / dx + Dl_22 * (c2_l[i + 1][j] - c2_l[i][j]) / dx)
            x2mins_l = (((1 - phi_out[i][j]) + (1 - phi_out[i - 1][j])) / 2) * (Dl_21 * (c1_l[i - 1][j] - c1_l[i][j]) / dx + Dl_22 * (c2_l[i - 1][j] - c2_l[i][j]) / dx)
            y2plus_l = (((1 - phi_out[i][j]) + (1 - phi_out[i][j + 1])) / 2) * (Dl_21 * (c1_l[i][j + 1] - c1_l[i][j]) / dx + Dl_22 * (c2_l[i][j + 1] - c2_l[i][j]) / dx)
            y2mins_l = (((1 - phi_out[i][j]) + (1 - phi_out[i][j - 1])) / 2) * (Dl_21 * (c1_l[i][j - 1] - c1_l[i][j]) / dx + Dl_22 * (c2_l[i][j - 1] - c2_l[i][j]) / dx)

            # part 1 in (17)
            part1_s = (x2plus_s + x2mins_s + y2plus_s + y2mins_s) / dx
            part1_l = (x2plus_l + x2mins_l + y2plus_l + y2mins_l) / dx

            # part 2 in (17)
            part2_s = P2 * phi_out[i][j] * (1 - phi_out[i][j]) * (mu2_l - mu2_s)
            part2_l = P2 * phi_out[i][j] * (1 - phi_out[i][j]) * (mu2_s - mu2_l)

            # part 3 in (17)
            part3_s = phi_out[i][j] * dphidt_opp * (c2_l[i][j] - c2_s[i][j])
            part3_l = (1 - phi_out[i][j]) * dphidt_ij * (c2_s[i][j] - c2_l[i][j])

            #dc/dt completed term
            if phi_out[i][j] < 1e-3: #if we are in liquid portion
                dc2sdt = 0.0
                dc2ldt = part1_l # (xplus_l + xmins_l + yplus_l + ymins_l) / dx
            elif phi_out[i][j] > 1-1e-3: #if we are in solid portion
                dc2sdt = part1_s # (xplus_s + xmins_s + yplus_s + ymins_s) / dx
                dc2ldt = 0.0
            else: #if(phi_out[i,j] >= 1e-9 && phi_out[i,j] <= 1-1e-9) #if we are at the interface solve both equations
                dc2sdt = (part1_s + part2_s - part3_s) / phi_out[i][j]
                dc2ldt = (part1_l + part2_l - part3_l) / (1-phi_out[i][j])
            # c2
            c2_s_out[i][j] = c2_s[i][j] + dt * dc2sdt
            c2_l_out[i][j] = c2_l[i][j] + dt * dc2ldt


            if c2_s_out[i][j] < 0.0:
                c2_s_out[i][j] = 0.0000001
            elif c2_s_out[i][j] > 1.0:
                c2_s_out[i][j] = 0.9999999

            if c2_l_out[i][j] < 0.0:
                c2_l_out[i][j] = 0.0000001
            elif c2_l_out[i][j] > 1.0:
                c2_l_out[i][j] = 0.9999999

            c2_out[i][j] = c2_s_out[i][j] * phi_out[i][j] + c2_l_out[i][j] * (1 - phi_out[i][j])

#BC
@cuda.jit
def TarrUpdate(fields,params,fields_out,timestep):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    Tarr = fields[7]
    beta = fields[16]
    Tarr_out = fields_out[7]
    beta_out = fields_out[16]

    dx = params[0]
    dt = params[1]
    Vs = params[5]
    G = params[6]
    initXpos = params[8]
    Ts = params[21]
    xoffs = params[22]

    t = timestep * dt
    for i in range(starty, Tarr.shape[0], stridey):
        for j in range(startx, Tarr.shape[1], stridex):

            #Tarr_out[i][j] = (Ts + G * ((j-1) * dx + xoffs - Vs * t))
            Tarr_out[i][j] = (Ts + G * ((j-1) * dx - initXpos * dx + xoffs - Vs * t))

            beta_out[i][j] = beta[i][j]
@cuda.jit
def initBC(fields):
    phi = fields[0]
    c1 = fields[1]
    c1_s = fields[2]
    c1_l = fields[3]
    c2 = fields[4]
    c2_s = fields[5]
    c2_l = fields[6]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty,phi.shape[0],stridey):
        for j in range(startx,phi.shape[1],stridex):
            ##### for ic initiate
            if i == 0 or i == phi.shape[0]:
                for j in range(phi.shape[1]):
                    phi[0][j] = phi[1][j]
                    phi[phi.shape[0]][j] = phi[phi.shape[0] - 1][j]
                    c1[0][j] = c1[1][j]
                    c2[0][j] = c2[1][j]
                    c1[phi.shape[0]][j] = c1[phi.shape[0] - 1][j]
                    c2[phi.shape[0]][j] = c2[phi.shape[0] - 1][j]
                    c1_l[0][j] = c1_l[1][j]
                    c2_l[0][j] = c2_l[1][j]
                    c1_l[phi.shape[0]][j] = c1_l[phi.shape[0] - 1][j]
                    c2_l[phi.shape[0]][j] = c2_l[phi.shape[0] - 1][j]
                    c1_s[0][j] = c1_s[1][j]
                    c2_s[0][j] = c2_s[1][j]
                    c1_s[phi.shape[0]][j] = c1_s[phi.shape[0] - 1][j]
                    c2_s[phi.shape[0]][j] = c2_s[phi.shape[0] - 1][j]
                   

            if j == 0 or j == phi.shape[1]:
                for i in range(phi.shape[0]):
                    phi[i][0] = phi[i][1]
                    phi[i][phi.shape[1]] = phi[i][phi.shape[1] - 1]
                    c1[i][0] = c1[i][1]
                    c2[i][0] = c2[i][1]
                    c1[i][phi.shape[1]] = c1[i][phi.shape[1] - 1]
                    c2[i][phi.shape[1]] = c2[i][phi.shape[1] - 1]
                    c1_s[i][0] = c1_s[i][1]
                    c2_s[i][0] = c2_s[i][1]
                    c1_s[i][phi.shape[1]] = c1_s[i][phi.shape[1] - 1]
                    c2_s[i][phi.shape[1]] = c2_s[i][phi.shape[1] - 1]
                    c1_l[i][0] = c1_l[i][1]
                    c2_l[i][0] = c2_l[i][1]
                    c1_l[i][phi.shape[1]] = c1_l[i][phi.shape[1] - 1]
                    c2_l[i][phi.shape[1]] = c2_l[i][phi.shape[1] - 1]

@cuda.jit
def PullBack(fields,params,fields_out):

    phi = fields[0]
    c1 = fields[1]
    c1_s = fields[2]
    c1_l = fields[3]

    c2 = fields[4]
    c2_s = fields[5]
    c2_l = fields[6]
    Tarr = fields[7]

    phi_out = fields_out[0]
    c1_out = fields_out[1]
    c1_s_out = fields_out[2]
    c1_l_out = fields_out[3]

    c2_out = fields_out[4]
    c2_s_out = fields_out[5]
    c2_l_out = fields_out[6]

    dx = params[0]
    dt = params[1]
    P1 = params[2]
    P2 = params[3]
    mu = params[4]  # ?? const

    initXpos = params[8]
    sigma = params[9]
    R = params[10]
    VM = params[11]
    ke_1 = params[12]  # kini
    ke_2 = params[13]
    c1_0 = params[14]
    c2_0 = params[15]
    me_1 = params[16]
    me_2 = params[17]
    k_an = params[18]
    v_an = params[19]
    Tn = params[20]
    Ts = params[21]
    kini_1 = params[12]
    kini_2 = params[13]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    M = phi.shape[1]

    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1]- 1, stridex):

            if j < M-1:
                phi_out[i][j] = phi_out[i][j + 1]
                c1_out[i][j] = c1_out[i][j+ 1]
                c1_l_out[i][j] = c1_l_out[i][j + 1]
                c1_s_out[i][j] = c1_s_out[i][j + 1]
                c2_out[i][j] = c2_out[i][j+ 1]
                c2_l_out[i][j] = c2_l_out[i][j + 1]
                c2_s_out[i][j] = c2_s_out[i][j + 1]

            else:
                phi_out[i][j] = 0
               
                c1_s_out[i][j] = c1_0 * kini_1 
                c1_l_out[i][j] = c1_0 
                c2_l_out[i][j] = c2_0 * kini_2 
                c2_l_out[i][j] = c2_0 



class ternary_FID_Nb(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL" #self._framework = "GPU_Parallel"

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        N = dim[0]
        M = dim[1]
        phi = np.zeros(dim)
        c1 = np.zeros(dim) # Al
        c1_s = np.ones(dim)
        c1_l = np.ones(dim)
        c2 = np.zeros(dim) # Nb
        c2_s = np.ones(dim)
        c2_l = np.ones(dim)

        Tarr = np.zeros(dim)
        np.random.seed(42)
        beta = np.random.rand(N, M).astype(np.float32)  #
        D_11_l =  np.zeros(dim)
        D_12_l = np.zeros(dim)
        D_21_l = np.zeros(dim)
        D_22_l = np.zeros(dim)
        D_11_s =  np.zeros(dim)
        D_12_s = np.zeros(dim)
        D_21_s = np.zeros(dim)
        D_22_s = np.zeros(dim)

        xi = self.user_data["xi"]
        dx = self.dx
        initXpos = self.user_data["initXpos"]
        Ts = self.user_data["Ts"]
        G = self.user_data["G"]
        kini_1 = self.user_data["kini_1"]
        kini_2 = self.user_data["kini_2"]

        c1_0 = self.user_data["c1_0"]
        c2_0 = self.user_data["c2_0"]

        # init Tarr
        # 创建 Tarr 的二维数组
        Tarr = Ts + G * (np.arange(M) * dx - initXpos * dx)  # 生成一维数组
        Tarr = np.tile(Tarr, (N, 1))  # 将 Tarr 一维数组复制成 N 行的二维数组

        #init phi
        posArr_h = initXpos * dx + (np.random.rand(N) - 0.5) * dx # 1D array
        posArr = np.zeros(N)
        posArr[:] = posArr_h

        x_p = np.arange(M) * dx  # 生成所有的 x_p
        phi = (1.0 - np. tanh((x_p - posArr[:, None]) / xi)) / 2  # 矢量化计算 phi

        phi = np.minimum(phi, 1.0)

        #init c_s and c_l , (init c1_s & c2_s = 1)
        c1_s = c1_s * c1_0 * kini_1
        c1_l = c1_l * c1_0
        c2_s = c2_s * c2_0 * kini_2
        c2_l = c2_l * c2_0
        c1 = c1_s * phi + c1_l * (1 - phi)
        c2 = c2_s * phi + c2_l * (1 - phi)

        self.add_field(phi, "phi",colormap=COLORMAP_PHASE)
        self.add_field(c1, "c1") #1
        self.add_field(c1_s,"c1_s") #2
        self.add_field(c1_l, "c1_l") #3
        self.add_field(c2, "c2")    #4
        self.add_field(c2_s,"c2_s") #5
        self.add_field(c2_l, "c2_l") #6

        self.add_field(Tarr,"Tarr") #7
        self.add_field(D_11_s, "D_11_s") #8
        self.add_field(D_12_s, "D_12_s")#9
        self.add_field(D_21_s, "D_21_s")#10
        self.add_field(D_22_s, "D_22_s")#11
        self.add_field(D_11_l, "D_11_l")#12
        self.add_field(D_12_l, "D_12_l")#13
        self.add_field(D_21_l, "D_21_l")#14
        self.add_field(D_22_l, "D_22_l")#15
        self.add_field(beta,"beta")
    def just_before_simulating(self):  #in simulation file
        super().just_before_simulating()

        params = []
        params.append(self.dx)              #0
        params.append(self.dt)              #1
        params.append(self.user_data["P1"]) #2
        params.append(self.user_data["P2"])  #3
        params.append(self.user_data["mu"]) #4
        params.append(self.user_data["Vs"]) #5
        params.append(self.user_data["G"])  #6
        params.append(self.user_data["dT"]) #7
        params.append(self.user_data["initXpos"]) #8
        params.append(self.user_data["sigma"])  #9
        params.append(self.user_data["R"])    #10
        params.append(self.user_data["VM"])    #11
        params.append(self.user_data["kini_1"]) #12
        params.append(self.user_data["kini_2"]) #13
        params.append(self.user_data["c1_0"])   #14
        params.append(self.user_data["c2_0"])  # 15
        params.append(self.user_data["me_1"])  #16
        params.append(self.user_data["me_2"])  #17
        params.append(self.user_data["k_an"])  #18
        params.append(self.user_data["v_an"])  #19
        params.append(self.user_data["Tn"])   #20
        params.append(self.user_data["Ts"])    #21
        
        params.append(self.user_data["xoffs"])#22
        params.append(self.user_data["xi"])#23

        params.append(self.user_data["Dl_11"])#24
        params.append(self.user_data["Dl_12"])#25
        params.append(self.user_data["Dl_22"])  # 26
        params.append(self.user_data["Ds_11"])  # 27
        params.append(self.user_data["Ds_12"])  # 28
        params.append(self.user_data["Ds_22"])  # 29
        params.append(self.user_data["Dl_21"]) # 30
        params.append(self.user_data["Ds_21"]) # 31
        
        self.user_data["params"] = np.array(params)
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])


    def simulation_loop(self):

        cuda.synchronize()

        solvePhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()
        
        solveC_const_D[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        cuda.synchronize()

        #solveC_Dij[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        cuda.synchronize()

        TarrUpdate[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device,self.time_step_counter -1)
        cuda.synchronize()
        
        tsave = self.user_data["tsave"]
        if (self.time_step_counter -1) % tsave                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    == 0:
            self.retrieve_fields_from_GPU()
            output_folder = self.user_data["save_path"]
            field_names = ['phi', 'c1', 'c2', 'tarr']
            os.makedirs(output_folder, exist_ok=True)
            for field, name in zip(self.fields, field_names):
                # Convert the field data to a DataFrame
                df = pd.DataFrame(field.data)
                # Construct the filename using the field name and the current time step counter
                filename = os.path.join(output_folder, f'{name}_step_{self.time_step_counter - 1}.csv')
                # Save the DataFrame to CSV without an index and without a header
                df.to_csv(filename, index=False, header=False)



        








