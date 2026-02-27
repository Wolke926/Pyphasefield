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
def BCC_L_Al_Nb_0(T):
    return -58005.3 + 13 * T
@numba.jit
def BCC_L_Al_Nb_1(T):
    return 26362.2 - 9.764 * T
@numba.jit
def BCC_L_Al_Nb_2(T):
    return 0
@numba.jit
def BCC_L_Al_Ti_0(T):
    return -128500 + 39 * T
@numba.jit
def BCC_L_Al_Ti_1(T):
    return 6000
@numba.jit
def BCC_L_Al_Ti_2(T):
    return 21200
@numba.jit
def BCC_L_Nb_Ti_0(T):
    return 8900
@numba.jit
def  BCC_L_Al_Nb_Ti_0(T):
    return -21442.5 - 13.647 * T
@numba.jit
def BCC_L_Al_Nb_Ti_1(T):
    return -3640.44 + 1.029 * T
@numba.jit
def BCC_L_Al_Nb_Ti_2(T):
    return -372108.8 + 212.18 * T
############################################
@numba.jit
def liq_L_Al_Nb_0(T):
    return -80470 + 23.476 * T
@numba.jit
def liq_L_Al_Nb_1(T):
    return -18176.2 + 14.282 * T
@numba.jit
def liq_L_Al_Nb_2(T):
    return 11868
@numba.jit
def liq_L_Al_Ti_0(T):
    return -108250 + 38 * T

@numba.jit
def liq_L_Al_Ti_1(T):
    return -6000 + 5 * T

@numba.jit
def liq_L_Al_Ti_2(T):
    return 15000
@numba.jit
def liq_L_Nb_Ti_0(T):
    return 3000

@numba.jit
def liq_L_Al_Nb_Ti_0(T):
    return - 49466.3 - 0.2377 * T

@numba.jit
def liq_L_Al_Nb_Ti_1(T):
    return 509335.7 - 239.65 * T

@numba.jit
def liq_L_Al_Nb_Ti_2(T):
    return  28183.0 - 3.303 * T
##########################################################
@numba.jit
def gibbsBCC_Al(T):
    return -1195.361 + 183.871153 * T - 31.748192 * T * safe_log(T) - 1230.622E25 * T**(-9)

@numba.jit
def gibbsliq_Al(T): 
    return -795.991 + 177.430178 * T - 31.748192 * T * safe_log(T)



@numba.jit
def gibbsliq_id(X, Y, T, R, VM):

    Z = 1 - X - Y
    gibbs_liq = (R * T/ VM) * (X * safe_log(X) + Y * safe_log(Y) + (1.0 - X - Y) * safe_log(1.0 - X -Y))
    gibbs_liq += (gibbsliq_Ti(T) * Z + gibbsliq_Nb(T) * Y + gibbsliq_Al(T) * X)/VM
    return gibbs_liq

@numba.jit
def gibbsliq_ex1(X, Y, T, R, VM):
    Z = 1 - X - Y
    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    term1 = X * Y *(L_Al_Nb_0 + L_Al_Nb_1 * (X-Y) + L_Al_Nb_2 * (X-Y)**2 )

    term2 = X * Z * (L_Al_Ti_0  + L_Al_Ti_1 * (X-Z) + L_Al_Ti_2 * (X-Z)**2 )
    #X-Z = 2*-Y-1
    term3 = Y * Z * L_Nb_Ti_0
    # Nb-Ti-2,Nb-Ti-1 =0,L_Nb_Ti_0 = 3000
    return (term1+term2+term3)/VM
@numba.jit
def gibbsliq_ex2(X, Y, T, R, VM):
    Z = 1 - X - Y
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)
    return  X * Y* Z * ( X * L_Al_Nb_Ti_0 + Y * L_Al_Nb_Ti_1 + Z * L_Al_Nb_Ti_2 ) /VM

@numba.jit
def f_L(X, Y, T, R, VM):
    gibbs_liq = gibbsliq_id(X, Y, T, R, VM) + gibbsliq_ex1(X, Y, T, R, VM) + gibbsliq_ex2(X, Y, T, R, VM)

    return gibbs_liq

@numba.jit
def Dgibbsliq_1(X, Y, T, R, VM):
    Z = 1-X - Y
    D_id = gibbsliq_Al(T) - gibbsliq_Ti(T) + R * T * (safe_log(X) - safe_log(Z)) #right

    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)

    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)

    term1 = Y * (L_Al_Nb_0 + L_Al_Nb_1 * (X - Y) + L_Al_Nb_2 * (X - Y) ** 2) + X * Y * (
                L_Al_Nb_1 + 2 * L_Al_Nb_2 * (X - Y))

    term2 = (Z - X) * (L_Al_Ti_0 + L_Al_Ti_1 * (X - Z) + L_Al_Ti_2 * (X - Z) ** 2) + (X * Z) * (
                2 * L_Al_Ti_1 + 4 * L_Al_Ti_2 * (X - Z))

    term3 = - Y * L_Nb_Ti_0

    Dex1_DX = term1 + term2 + term3

    Dex2_DX = (Y - 2 * X * Y - Y ** 2) * (X * L_Al_Nb_Ti_0 + Y * L_Al_Nb_Ti_1 + Z * L_Al_Nb_Ti_2) + (X * Y * Z) * (
                L_Al_Nb_Ti_0 - L_Al_Nb_Ti_2)
   
    LAlNbD1 = L_Al_Nb_0 * Y + L_Al_Nb_1 * (X * Y - Y ** 2)
    LAlTiD1 = (Z - X) * L_Al_Ti_0
    LNbTiD1 = - Y * L_Nb_Ti_0
    #return (D_id + LAlNbD1 + LAlTiD1+LNbTiD1)/ VM
    return (D_id + Dex1_DX + Dex2_DX) / VM

@numba.jit
def Dgibbsliq_2(X, Y, T, R, VM):
    Z = 1-X-Y
    D_id_DY = gibbsliq_Nb(T) - gibbsliq_Ti(T) + R*T *(safe_log(Y) - safe_log(Z))

    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)

    term1 = X*( L_Al_Nb_0 + L_Al_Nb_1 * (X-Y) + L_Al_Nb_2*(X-Y)**2) + X*Y*(- L_Al_Nb_1 - 2*L_Al_Nb_2*(X-Y))
    term2 = -X*(L_Al_Ti_0 + L_Al_Ti_1*(X-Z) + L_Al_Ti_2 * (X-Z)**2) + (X*Z)*(L_Al_Ti_1+2*L_Al_Ti_2 * (X-Z))
    term3 = (1- X - 2*Y)* L_Nb_Ti_0
    Dex1_DY = term1 + term2 + term3

    Dex2_DY = (X - 2 * X * Y - X ** 2) * (X * L_Al_Nb_Ti_0 + Y * L_Al_Nb_Ti_1 + Z * L_Al_Nb_Ti_2) + (X * Y *Z) * (L_Al_Nb_Ti_1 - L_Al_Nb_Ti_2)
    #return (D_id_DY + Dex1_DY + Dex2_DY)/VM
    LAlNbD2 = X*( L_Al_Nb_0 + L_Al_Nb_1 * (X-Y) -Y*L_Al_Nb_1)
    LAlTiD2 =-X*(L_Al_Ti_0 + L_Al_Ti_1*(X-Z)) + (X*Z)*(L_Al_Ti_1)
    LNbTiD2  =(1- X - 2*Y)* L_Nb_Ti_0
    #return (D_id_DY + LAlNbD2 + LAlTiD2+LNbTiD2)/ VM
    return (D_id_DY + Dex1_DY + Dex2_DY)/VM
###################################################################


@numba.jit
def gibbsBCC_id(X, Y, T, R, VM):
    Z = 1 - X - Y
    gibbsBCC_id = (R * T / VM) * (X * safe_log(X) + Y * safe_log(Y) + (1.0 - X - Y) * safe_log(1.0 - X - Y))
    gibbsBCC_id = gibbsBCC_id + (gibbsBCC_Ti(T) * Z + gibbsBCC_Nb(T) * Y + gibbsBCC_Al(T) * X) / VM
    return gibbsBCC_id

@numba.jit
def gibbsBCC_ex1(X, Y, T, R, VM):
    Z = 1 - X - Y
    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
   
    term1 = X * Y * (L_Al_Nb_0 + L_Al_Nb_1 * (X - Y) + L_Al_Nb_2 * (X - Y) ** 2)
    term2 = X * Z * (L_Al_Ti_0 + L_Al_Ti_1 * (X - Z) + L_Al_Ti_2 * (X - Z) ** 2)  # X-Z = 2*-Y-1
    term3 = Y * Z * L_Nb_Ti_0

    return (term1 + term2 + term3)/VM


@numba.jit
def gibbsBCC_ex2(X, Y, T, R, VM):
    Z = 1 - X - Y
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)
    return X * Y * Z * (1/VM) * (X * L_Al_Nb_Ti_0 + Y * L_Al_Nb_Ti_1 + Z * L_Al_Nb_Ti_2)
@numba.jit
def f_S(X, Y, T, R, VM):
    return  gibbsBCC_id(X, Y, T, R, VM)  + gibbsBCC_ex1(X, Y, T, R, VM) + gibbsBCC_ex2(X, Y, T, R, VM)




#################################################################################################


#B14
@numba.jit
def DgibbsBCC_1(X,Y,T,R,VM):

    Z =  1 - X - Y

    D_id = gibbsBCC_Al(T) - gibbsBCC_Ti(T) + R * T * (safe_log(X) - safe_log(Z))

    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)

    term1 = Y * (L_Al_Nb_0 + L_Al_Nb_1 * (X - Y) + L_Al_Nb_2 * (X - Y) ** 2) + X * Y * (L_Al_Nb_1 + 2 * L_Al_Nb_2 * (X - Y))
    term2 = (Z - X) * (L_Al_Ti_0 + L_Al_Ti_1 * (X - Z) + L_Al_Ti_2 * (X - Z) ** 2) + (X * Z) * (2 * L_Al_Ti_1 + 4 * L_Al_Ti_2 * (X - Z))
    term3 = - Y * L_Nb_Ti_0
    Dex1_DX = term1 + term2 + term3

    Dex2_DX = (Y - 2 * X * Y - Y ** 2) * (X * L_Al_Nb_Ti_0 + Y * L_Al_Nb_Ti_1 + Z * L_Al_Nb_Ti_2) + (X * Y *Z) * (L_Al_Nb_Ti_0 - L_Al_Nb_Ti_2)

    return (D_id + Dex1_DX + Dex2_DX)/VM


@numba.jit
def DgibbsBCC_2(X,Y,T,R,VM):
    Z =  1 - X - Y
    D_id_DY = gibbsBCC_Nb(T) - gibbsBCC_Ti(T) + R * T * (safe_log(Y) - safe_log(Z))

    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)

    term1 = X * (L_Al_Nb_0 + L_Al_Nb_1 * (X - Y) + L_Al_Nb_2 * (X - Y) ** 2) + X * Y*(- L_Al_Nb_1 - 2 * L_Al_Nb_2 * (X - Y))
    term2 = -X*(L_Al_Ti_0 + L_Al_Ti_1 * (X - Z) + L_Al_Ti_2 * (X - Z) ** 2) + (X * Z) * (L_Al_Ti_1 + 2 * L_Al_Ti_2 * (X - Z))
    term3 = (1- X - 2 * Y) * L_Nb_Ti_0
    Dex1_DY = term1 + term2 + term3

    Dex2_DY = (X - 2 * X * Y - X ** 2) * (X * L_Al_Nb_Ti_0 + Y * L_Al_Nb_Ti_1 + Z * L_Al_Nb_Ti_2) + ( X * Y *Z) * (L_Al_Nb_Ti_1 - L_Al_Nb_Ti_2)

    return (D_id_DY + Dex1_DY + Dex2_DY)/VM

#################################################################

@numba.jit
def gibbsBCC_Ti(T): #?
    
    #return 5511.037 + 66.976538*T - 14.9466 * T * safe_log(T) - 8.1465E-3 * T**2 + 0.202715E-6 * T**3 -1477660/T  #BCTA5/CBCCA12 1155-1941
    return 6667.385+105.366379*T-22.3771*T*safe_log(T)+.00121707*T**2-8.4534E-07*T**3 -2002750/T   #BCCA2 1115-1941
    #return -118526.786+638.706871*T - 87.2182461*T*safe_log(T)+ 8.204849E-3*T**2-0.304747e-6*T**3+36699805/T #FCCa1 1941-4000

    #return 26483.26 - 182.426271 * T + 19.0900905 * T * safe_log(T) - 22.00832E-3*T**2 + 1.228863E-6 * T**3 + 1400501/T # bcca2 > 1941
    #return -119924.586 + 638.806871 *T - 87.2182461*T * safe_log(T)+8.20484E-3 *T**2 - 0.304747E-6* T**3 + 36699805/T #bcta5/cbcca12 > 1941


@numba.jit    
def gibbsliq_Ti(T): #?
    return -19887.006 + 298.7367 * T - 46.29 * T * safe_log(T)
    #return 369519.198 - 2554.0225*T + 342.059267 * T * safe_log(T) - 163.409355E-3 *T**2+12.457117E-6*T**3 - 67034516/T # <1941

@numba.jit    
def gibbsBCC_Nb(T): #?
    return -8519.353 + 142.045474 * T - 26.4711 * T * safe_log(T) + 0.203475E-3 * T**2 - 0.35012E-6 * T**3 + 93399/T # < 2750K
    #return -37669.3 + 271.720843 * T - 41.77 * T * safe_log(T) + 1528.238E29*T**(-9)
    # return 4680.647+143.745475 *T -26.4711*T* safe_log(T)+ 0.203475E-3*T**2 - 0.35012E-6*T**3 + 93399/T
    # return -24169.3 + 273.420843*T - 41.77*T* safe_log(T) + 1528.238E29* T**(-9) # > 2750K

@numba.jit
def gibbsliq_Nb(T):

    return 21262.202 + 131.229057*T - 26.4711*T * safe_log(T)+0.203475E-3*T**2 - 0.350119E-6*T**3 + 93399/T - 3.06098E-23*T**7 # T<2750
    #return -7499.398+260.756148*T- 41.77* T * safe_log(T)
 #######################################################

@numba.jit
def DgibbsBCC_3_YZ(X,Y,T,R,VM):
    Z =1-X-Y
    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)
    D_id_DZ = gibbsBCC_Ti(T) - gibbsBCC_Al(T) + R * T * (math.log(Z) - math.log(X))
    term1 =Y*X * (-2*L_Al_Nb_2*(X-Y)-L_Al_Nb_1)-Y*(L_Al_Nb_1*(X-Y)+L_Al_Nb_2*(X-Y)**2+L_Al_Nb_0)
    term2 = (- 4* L_Al_Ti_2 *(X-Z)- 2 * L_Al_Ti_1) * X *Z - (L_Al_Ti_1*(X-Z)+L_Al_Ti_2 * (X-Z)**2+ L_Al_Ti_0) *Z+\
            (L_Al_Ti_1 * ( X - Z)+ L_Al_Ti_2 * (Z - X)**2+ L_Al_Ti_0) * X
    term3 = Y * L_Nb_Ti_0
    Dex1_DZ = term1+term2+term3

    Dex2_DZ = (X*Y-Y*Z)* (L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)+(L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*Y*X*Z
    return (D_id_DZ + Dex1_DZ + Dex2_DZ) / VM

@numba.jit
def Dgibbsliq_3_YZ(X,Y,T,R,VM):
    Z = 1 - X - Y

    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)

    D_id_DZ = gibbsliq_Ti(T) - gibbsliq_Al(T) + R * T * (math.log(Z) - math.log(X))
    term1 =Y*(-2*L_Al_Nb_2*(- Z-2*Y+1)-L_Al_Nb_1)*(-Z-Y+1)-Y*(L_Al_Nb_1*(-Z - 2*Y+1)+L_Al_Nb_2*(-Z-2*Y+1)**2+L_Al_Nb_0)
    term2 = (- 4* L_Al_Ti_2 *(X-Z)- 2 * L_Al_Ti_1) * X *Z - (L_Al_Ti_1*(X-Z)+L_Al_Ti_2 * (X-Z)**2+ L_Al_Ti_0) *Z+\
            (L_Al_Ti_1 * ( X - Z)+ L_Al_Ti_2 * (Z - X)**2+ L_Al_Ti_0) * X
    term3 = Y * L_Nb_Ti_0
    Dex1_DZ = term1+term2+term3

    Dex2_DZ = (X*Y-Y*Z)* (L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)+(L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*Y*X*Z

    return (D_id_DZ + Dex1_DZ + Dex2_DZ) / VM

@numba.jit
def DgibbsBCC_3_XZ(X,Y,T,R,VM):
    Z =1-X-Y
    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)

    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)

    D_id_DZ = gibbsBCC_Ti(T) - gibbsBCC_Nb(T) + R * T * (math.log(Z) - math.log(Y))
    term1 = X*Y * (2*L_Al_Nb_2* (Z+2*X-1)+L_Al_Nb_1)- X*(L_Al_Nb_2*(Z+2*X-1)**2+L_Al_Nb_1* (Z+2*X-1)+L_Al_Nb_0)
    term2 = X*(- 2*L_Al_Ti_2*(X-Z)-L_Al_Ti_1)* Z+X*(L_Al_Ti_1* (X-Z)+L_Al_Ti_2*(X-Z)**2+L_Al_Ti_0)
    term3 = L_Nb_Ti_0 * (-Z-X+1)-L_Nb_Ti_0*Z
    Dex1_DZ = term1 + term2 + term3
    Dex2_DZ =(X*Y-X*Z)*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_1*Y+L_Al_Nb_Ti_0*X)+(L_Al_Nb_Ti_2-L_Al_Nb_Ti_1)*X*Y*Z
    return (D_id_DZ + Dex1_DZ + Dex2_DZ) / VM
@numba.jit
def Dgibbsliq_3_XZ(X,Y,T,R,VM):
    Z = 1 - X - Y

    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)

    D_id_DZ = gibbsliq_Ti(T) - gibbsliq_Nb(T) + R * T * (math.log(Z) - math.log(Y))
    term1 = X*Y * (2*L_Al_Nb_2* (Z+2*X-1)+L_Al_Nb_1)- X*(L_Al_Nb_2*(Z+2*X-1)**2+L_Al_Nb_1* (Z+2*X-1)+L_Al_Nb_0)
    term2 = X*(- 2*L_Al_Ti_2*(X-Z)-L_Al_Ti_1)* Z+X*(L_Al_Ti_1* (X-Z)+L_Al_Ti_2*(X-Z)**2+L_Al_Ti_0)
    term3 = L_Nb_Ti_0 * (-Z-X+1)-L_Nb_Ti_0*Z
    Dex1_DZ = term1 + term2 + term3
    Dex2_DZ =(X*Y-X*Z)*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_1*Y+L_Al_Nb_Ti_0*X)+(L_Al_Nb_Ti_2-L_Al_Nb_Ti_1)*X*Y*Z
    return (D_id_DZ + Dex1_DZ + Dex2_DZ) / VM
@numba.jit
def DDgibbsBCC_11(X,Y,T,R,VM):
    Z = 1 - X - Y
    DDid_11 = R*T * (1-Y)/(X * Z)

    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)

    DDterm1_11 = 2 * Y * (3 * L_Al_Nb_2 * X - 2 * L_Al_Nb_2 * Y + L_Al_Nb_1)
    DDterm2_11 = -48 * L_Al_Ti_2 * X ** 2 - 48 * Y * X * L_Al_Ti_2 + 48 * X * L_Al_Ti_2 - 12 * L_Al_Ti_1 * X - \
                 10 * L_Al_Ti_2 * Y ** 2 - 6 * L_Al_Ti_1 * Y + 20 * L_Al_Ti_2 * Y - 10 * L_Al_Ti_2 + 6 * L_Al_Ti_1 - 2 * L_Al_Ti_0
    DDterm3_11 = 0

    DDex1_11 = DDterm1_11 + DDterm2_11 + DDterm3_11

    DDex2_11 = 2*Y * ((3*L_Al_Nb_Ti_2-3*L_Al_Nb_Ti_0)*X + (2*L_Al_Nb_Ti_2-L_Al_Nb_Ti_1 - L_Al_Nb_Ti_0)*Y - 2 *L_Al_Nb_Ti_2 + L_Al_Nb_Ti_0)
    return (DDid_11 + DDex1_11 + DDex2_11)/VM
@numba.jit
def DDgibbsliq_11(X,Y,T,R,VM):
    Z = 1 - X - Y
    DDid_11 = R*T * (1-Y)/(X * Z)
    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)

    DDterm1_11 = 2 * Y * (3 * L_Al_Nb_2 *X - 2* L_Al_Nb_2 * Y + L_Al_Nb_1)
    DDterm2_11 = -48 *L_Al_Ti_2*X**2 - 48*Y*X * L_Al_Ti_2 + 48*X* L_Al_Ti_2 - 12*L_Al_Ti_1*X - \
                 10*L_Al_Ti_2*Y**2 - 6 *L_Al_Ti_1 * Y + 20*L_Al_Ti_2*Y - 10 *L_Al_Ti_2 + 6*L_Al_Ti_1 - 2 * L_Al_Ti_0
    DDterm3_11 = 0

    DDex1_11 = DDterm1_11 + DDterm2_11 + DDterm3_11

    DDex2_11 = 2*Y * ((3* L_Al_Nb_Ti_2 -3*L_Al_Nb_Ti_0)*X + (2* L_Al_Nb_Ti_2-L_Al_Nb_Ti_1 - L_Al_Nb_Ti_0)*Y - 2 *L_Al_Nb_Ti_2 + L_Al_Nb_Ti_0)
    return (DDid_11 + DDex1_11 + DDex2_11)/VM

@numba.jit
def DDgibbsBCC_12(X,Y,T,R,VM):
    Z = 1 - X - Y

    DDid_12 = - R * T / Z
    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)
    DDterm1_12 = (-2*L_Al_Nb_2*(X-Y)-L_Al_Nb_1)*Y-2*L_Al_Nb_2*X*Y+L_Al_Nb_1*(X-Y)+X*(2*L_Al_Nb_2*(X-Y)+L_Al_Nb_1)+L_Al_Nb_2*(X-Y)**2+L_Al_Nb_0
    DDterm2_12 = -(L_Al_Ti_0+L_Al_Ti_1*(Y+2*X-1)+L_Al_Ti_2*(Y+2*X-1)**2) + (Y+2*X-1)*(2*L_Al_Ti_2-2*L_Al_Ti_2*Y - 8*X*L_Al_Ti_2)+\
                 (1-Y-4*X)*L_Al_Ti_1 + 4* (X - X*Y -X**2)*L_Al_Ti_2
    DDterm3_12 = - L_Nb_Ti_0
    DDex1_12 = DDterm1_12 + DDterm2_12+DDterm3_12

    DDex2_12 = (3 * L_Al_Nb_Ti_2 - 3 * L_Al_Nb_Ti_1) * Y ** 2 + (( 8 * L_Al_Nb_Ti_2 - 4 * L_Al_Nb_Ti_1 - 4 * L_Al_Nb_Ti_0) * X - 4 * L_Al_Nb_Ti_2 + 2 * L_Al_Nb_Ti_1) * Y + \
               (3 * L_Al_Nb_Ti_2 - 3 * L_Al_Nb_Ti_0) * X ** 2 + (2 * L_Al_Nb_Ti_0 - 4 * L_Al_Nb_Ti_2) * X + L_Al_Nb_Ti_2
    return (DDid_12 + DDex1_12 + DDex2_12)/VM
@numba.jit
def DDgibbsliq_12(X, Y, T, R, VM):
    Z = 1 - X - Y
    DDid_12 = R * T / Z
    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)

    DDterm1_12 = (-2*L_Al_Nb_2*(X-Y)-L_Al_Nb_1)*Y-2*L_Al_Nb_2*X*Y+L_Al_Nb_1*(X-Y)+X*(2*L_Al_Nb_2*(X-Y)+L_Al_Nb_1)+L_Al_Nb_2*(X-Y)**2+L_Al_Nb_0
    DDterm2_12 = -(L_Al_Ti_0+L_Al_Ti_1*(Y+2*X-1)+L_Al_Ti_2*(Y+2*X-1)**2) + (Y+2*X-1)*(2*L_Al_Ti_2-2*L_Al_Ti_2*Y - 8*X*L_Al_Ti_2)+\
                 (1-Y-4*X)*L_Al_Ti_1 + 4* (X - X*Y -X**2)*L_Al_Ti_2
    DDterm3_12 = - L_Nb_Ti_0
    DDex1_12 = DDterm1_12 + DDterm2_12+DDterm3_12

    DDex2_12 = (3*L_Al_Nb_Ti_2-3*L_Al_Nb_Ti_1)*Y**2 + ((8*L_Al_Nb_Ti_2-4*L_Al_Nb_Ti_1-4*L_Al_Nb_Ti_0)*X - 4*L_Al_Nb_Ti_2+2*L_Al_Nb_Ti_1)*Y + \
               (3*L_Al_Nb_Ti_2-3*L_Al_Nb_Ti_0)*X**2 + (2*L_Al_Nb_Ti_0-4*L_Al_Nb_Ti_2)*X + L_Al_Nb_Ti_2
    return (DDid_12 + DDex1_12 + DDex2_12)/VM
@numba.jit
def DDgibbsBCC_13(X, Y, T, R, VM):
    Z = 1 - X - Y
    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)

    DDid_13 = R * T * (- 1/Z - 1/X)

    term1 = -2*L_Al_Nb_2 * Y*(1-Y-Z) - Y* (2*L_Al_Nb_2*(-Z-2*Y+1)+L_Al_Nb_1)+Y*(- 2* L_Al_Nb_2* (-Z-2*Y+1)-L_Al_Nb_1)
    term2 = (-4*L_Al_Ti_2*(-2*Z-Y+1)-2*L_Al_Ti_1)*(2*Z+Y-1)-8*L_Al_Ti_2*X*Z-\
            (4*L_Al_Ti_2*(-2*Z-Y+1)+2*L_Al_Ti_1)*Z+(4*L_Al_Ti_2*(-2*Z-Y+1)+2*L_Al_Ti_1)*X+2*(L_Al_Ti_1*(-2*Z-Y+1)+L_Al_Ti_2*(-2*Z-Y+1)*2+L_Al_Ti_0)
    term3 = - L_Nb_Ti_0
    DDex1_13 = term1 + term2 + term3
    DDex2_13 =(L_Al_Nb_Ti_0-L_Al_Nb_Ti_1)*(X*Y-X*Z)+(-2*Z-2*X+1)*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_1*Y+L_Al_Nb_Ti_0*X)+(L_Al_Nb_Ti_2-L_Al_Nb_Ti_1)*Y*Z-(L_Al_Nb_Ti_2-L_Al_Nb_Ti_1)*X*Z
    return (DDid_13 + DDex1_13 + DDex2_13) / VM
@numba.jit
def DDgibbsliq_13(X, Y, T, R, VM):
    Z = 1 - X - Y
    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)

    DDid_13 = R * T * ( - 1/Z - 1/X)

    term1 = -2*L_Al_Nb_2 * Y* (-Z-Y+1)- Y* (2*L_Al_Nb_2*(-Z-2*Y+1)+L_Al_Nb_1)+Y*(-2*L_Al_Nb_2*(-Z-2*Y+1)-L_Al_Nb_1)
    term2 = (-4*L_Al_Ti_2*(-2*Z-Y+1)-2*L_Al_Ti_1)*(2*Z+Y-1)-8*L_Al_Ti_2*X*Z-\
            (4*L_Al_Ti_2*(-2*Z-Y+1)+2*L_Al_Ti_1)*Z+(4*L_Al_Ti_2*(-2*Z-Y+1)+2*L_Al_Ti_1)*X+2*(L_Al_Ti_1*(-2*Z-Y+1)+L_Al_Ti_2*(-2*Z-Y+1)*2+L_Al_Ti_0)
    term3 = 0
    DDex1_13 = term1 + term2 + term3
    DDex2_13 =2*Y*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)+(L_Al_Nb_Ti_2-L_Al_Nb_Ti_0)*Y*(2*Z+Y-1)-\
              (L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*Y*Z+(L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*Y*X
    return (DDid_13 + DDex1_13 + DDex2_13) / VM
@numba.jit
def DDgibbsBCC_21(X, Y, T, R, VM):
    Z = 1-Y-X

    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)

    DDid_21 = R * T / Z
    term1 = L_Al_Nb_0 + 3 * L_Al_Nb_2 * X ** 2 + 2 * L_Al_Nb_1 * X - 8 * L_Al_Nb_2 * X * Y + 3 * L_Al_Nb_2 * Y ** 2
    term2 = -(L_Al_Ti_0+L_Al_Ti_1*(Y+2*X-1)+L_Al_Ti_2*(Y+2*X-1)**2) + (Y+2*X-1)*(2*L_Al_Ti_2-2*L_Al_Ti_2*Y - 8*X*L_Al_Ti_2)+\
                 (1-Y-4*X)*L_Al_Ti_1 + 4* (X - X*Y -X**2)*L_Al_Ti_2
    term3 = -L_Nb_Ti_0
    DDex1_21 = term1+term2+term3
    DDex2_21 = (L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*(-2*X*Y-X**2+X)+(-2*Y-2*X+1)*(L_Al_Nb_Ti_1*Y+L_Al_Nb_Ti_2*(-Y-X+1)+\
                L_Al_Nb_Ti_0*X)+(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*(-Y-X+1)*Y-(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*X*Y
    return (DDid_21+ DDex1_21+DDex2_21)/VM
@numba.jit
def DDgibbsliq_21(X, Y, T, R, VM):
    Z = 1 - X - Y

    DDid_21 = R * T /Z

    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)
    term1 = L_Al_Nb_0+3*L_Al_Nb_2*X**2 + 2*L_Al_Nb_1*X - 8*L_Al_Nb_2*X*Y + 3*L_Al_Nb_2*Y**2
    term2 = -(L_Al_Ti_0+L_Al_Ti_1*(Y+2*X-1)+L_Al_Ti_2*(Y+2*X-1)**2) + (Y+2*X-1)*(2*L_Al_Ti_2-2*L_Al_Ti_2*Y - 8*X*L_Al_Ti_2)+\
                 (1-Y-4*X)*L_Al_Ti_1 + 4* (X - X*Y -X**2)*L_Al_Ti_2
    term3 = -L_Nb_Ti_0

    DDex1_21 = term1+ term2+ term3
    DDex2_21 =(L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*(-2*X*Y-X**2+X)+(-2*Y-2*X+1)*(L_Al_Nb_Ti_1*Y+L_Al_Nb_Ti_2*(-Y-X+1)+\
                L_Al_Nb_Ti_0*X)+(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*(-Y-X+1)*Y-(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*X*Y
    return (DDid_21+ DDex1_21+DDex2_21)/VM
@numba.jit
def DDgibbsBCC_22(X, Y, T, R, VM):
    Z = 1 - X - Y

    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)
    DDid_22 = R*T *(1/Y + 1/Z)
    term1 = 2*L_Al_Nb_2*X*Y + X*(L_Al_Nb_1 - 2*L_Al_Nb_2*(X-Y)) + X*(-2*L_Al_Nb_2*(X-Y)-L_Al_Nb_1)
    term2 = 2*L_Al_Ti_2*X*(-Y-X-1) - 2*X*(2 * L_Al_Ti_2*(Y+2*X-1)+L_Al_Ti_1)
    term3 = -2* L_Nb_Ti_0
    DDex1_22 = term1+ term2+ term3
    DDex2_22 = (L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*(-2*X*Y - X**2 + X)-2*X*(L_Al_Nb_Ti_1*Y+L_Al_Nb_Ti_2 * Z +L_Al_Nb_Ti_0*X)-(L_Al_Nb_Ti_1- L_Al_Nb_Ti_2)*X*Y+(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*X*Z
    return (DDid_22 + DDex1_22 + DDex2_22)/VM
@numba.jit
def DDgibbsliq_22(X, Y, T, R, VM):
    Z = 1 - X- Y
    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)
    DDid_22 = R * T * (1 / Y + 1 / Z)
    term1 = 2 * L_Al_Nb_2 * X * Y + X * (L_Al_Nb_1 - 2 * L_Al_Nb_2 * (X - Y)) + X * (-2 * L_Al_Nb_2 * (X - Y) - L_Al_Nb_1)
    term2 = 2 * L_Al_Ti_2 * X * (-Y - X - 1) - 2 * X * (2 * L_Al_Ti_2 * (Y + 2 * X - 1) + L_Al_Ti_1)
    term3 = -2 * L_Nb_Ti_0
    DDex1_22 = term1 + term2 + term3
    DDex2_22 = (L_Al_Nb_Ti_1 - L_Al_Nb_Ti_2) * (-2 * X * Y - X ** 2 + X) - 2 * X * (L_Al_Nb_Ti_1 * Y + L_Al_Nb_Ti_2 * Z + L_Al_Nb_Ti_0 * X) - (L_Al_Nb_Ti_1 - L_Al_Nb_Ti_2) * X * Y + (L_Al_Nb_Ti_1 - L_Al_Nb_Ti_2) * X * Z
    return (DDid_22 + DDex1_22 + DDex2_22)/VM
@numba.jit
def DDgibbsBCC_23(X, Y, T, R, VM):
    Z = 1 - X - Y

    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)

    DDid_23 = - R * T /Z
    term1 =(-2*L_Al_Nb_2*(-Z-2*Y+1)-L_Al_Nb_1)*X+2*L_Al_Nb_2*Y*X-L_Al_Nb_1*(-Z-2*Y+1)-\
           Y*(-2*L_Al_Nb_2*(-Z-2*Y+1)-L_Al_Nb_1)-L_Al_Nb_2*(-Z-2*Y+1)**2-L_Al_Nb_0
    term2 = (-2*L_Al_Ti_2*(-Z-2*Y+1)-L_Al_Ti_1)*(Z+Y-1)-4*L_Al_Ti_2*X*Z-(2*L_Al_Ti_2*(-2*Z-Y+1)+L_Al_Ti_1)* Z+(2*L_Al_Ti_2*(-2*Z-Y+1)+L_Al_Ti_1)*X\
            +L_Al_Ti_1*(-Z-2*Y+1)+L_Al_Ti_2*(-Z-2*Y+1)**2+L_Al_Ti_0
    term3 = 0
    DDex1_23 = term1 + term2 + term3
    DDex2_23 =-(Z-Y)*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)+X*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)+\
        (L_Al_Nb_Ti_2-L_Al_Nb_Ti_0)*X*(Z-Y)-(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*Y*Z+(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*Y*X
    return (DDid_23 + DDex1_23 + DDex2_23)/VM
@numba.jit
def DDgibbsliq_23(X, Y, T, R, VM):
    Z = 1 - X - Y

    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)


    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)
    DDid_23 = - R * T /Z
    term1 =(-2*L_Al_Nb_2*(-Z-2*Y+1)-L_Al_Nb_1)*X+2*L_Al_Nb_2*Y*X-L_Al_Nb_1*(-Z-2*Y+1)-\
           Y*(-2*L_Al_Nb_2*(-Z-2*Y+1)-L_Al_Nb_1)-L_Al_Nb_2*(-Z-2*Y+1)**2-L_Al_Nb_0
    term2 = (-2*L_Al_Ti_2*(-Z-2*Y+1)-L_Al_Ti_1)*(Z+Y-1)-4*L_Al_Ti_2*X*Z-(2*L_Al_Ti_2*(-2*Z-Y+1)+L_Al_Ti_1)* Z+(2*L_Al_Ti_2*(-2*Z-Y+1)+L_Al_Ti_1)*X\
            +L_Al_Ti_1*(-Z-2*Y+1)+L_Al_Ti_2*(-Z-2*Y+1)**2+L_Al_Ti_0
    term3 = 0
    DDex1_23 = term1 + term2 + term3
    DDex2_23 =-(Z-Y)*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)+X*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)+\
        (L_Al_Nb_Ti_2-L_Al_Nb_Ti_0)*X*(Z-Y)-(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*Y*Z+(L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*Y*X
    return (DDid_23 + DDex1_23 + DDex2_23)/VM
@numba.jit
def DDgibbsBCC_31(X, Y, T, R, VM):
    Z = 1-X-Y
    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)
    DDid_31 = R*T/(-Z-X+1)

    term1 = -L_Al_Nb_2*(Z+2*X-1)**2-X*(4*L_Al_Nb_2*(Z+2*X-1)+2*L_Al_Nb_1)+Y*(2*L_Al_Nb_2*(Z+2*X-1)+L_Al_Nb_1)-\
            X*(2*L_Al_Nb_2*(Z+2*X-1)+L_Al_Nb_1)-L_Al_Nb_1*(Z+2*X-1)+4*L_Al_Nb_2*X*Y-L_Al_Nb_0
    term2 =(-2*L_Al_Ti_2*(X-Z)-L_Al_Ti_1)*Z-2*L_Al_Ti_2*X*Z+L_Al_Ti_1*(X-Z)+X*(2*L_Al_Ti_2*(X-Z)+L_Al_Ti_1)+L_Al_Ti_2*(X-Z)**2+L_Al_Ti_0
    term3 = -L_Nb_Ti_0
    DDex1_31 = term1 + term2 + term3
    DDex2_31 = (L_Al_Nb_Ti_0-L_Al_Nb_Ti_1)*(X*Y-X*Z)+(-2*Z-2*X+1)*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_1*Y+L_Al_Nb_Ti_0*X)+(L_Al_Nb_Ti_2-L_Al_Nb_Ti_1)*Y*Z-(L_Al_Nb_Ti_2 - L_Al_Nb_Ti_1)*X*Z
    return (DDid_31 + DDex1_31 + DDex2_31)/VM
@numba.jit
def DDgibbsliq_31(X, Y, T, R, VM):
    Z = 1-X-Y
    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)

    DDid_31 = R*T/Y
    term1 = -L_Al_Nb_2 * (Z + 2 * X - 1) ** 2 - X * (4 * L_Al_Nb_2 * (Z + 2 * X - 1) + 2 * L_Al_Nb_1) + Y * (
                2 * L_Al_Nb_2 * (Z + 2 * X - 1) + L_Al_Nb_1) - \
            X * (2 * L_Al_Nb_2 * (Z + 2 * X - 1) + L_Al_Nb_1) - L_Al_Nb_1 * (
                        Z + 2 * X - 1) + 4 * L_Al_Nb_2 * X * Y - L_Al_Nb_0
    term2 = (-2 * L_Al_Ti_2 * (X - Z) - L_Al_Ti_1) * Z - 2 * L_Al_Ti_2 * X * Z + L_Al_Ti_1 * (X - Z) + X * (
                2 * L_Al_Ti_2 * (X - Z) + L_Al_Ti_1) + L_Al_Ti_2 * (X - Z) ** 2 + L_Al_Ti_0
    term3 = -L_Nb_Ti_0
    DDex1_31 = term1 + term2 + term3
    DDex2_31 = (L_Al_Nb_Ti_0 - L_Al_Nb_Ti_1)*(X * Y - X * Z) + (-2 * Z - 2 * X + 1)*(
        L_Al_Nb_Ti_2 * Z + L_Al_Nb_Ti_1 * Y+L_Al_Nb_Ti_0 * X) + (L_Al_Nb_Ti_2 - L_Al_Nb_Ti_1) * Y * Z - (
                           L_Al_Nb_Ti_2 - L_Al_Nb_Ti_1) * X * Z
    return (DDid_31 + DDex1_31 + DDex2_31) / VM
@numba.jit
def DDgibbsBCC_32(X, Y, T, R, VM):
    Z = 1-X-Y
    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)
    DDid_32 = - R*T/Z
    term1 =-L_Al_Nb_2*X**2+X*(-2*L_Al_Nb_2*X-L_Al_Nb_1)-L_Al_Nb_1*X-L_Al_Nb_0
    term2 = L_Al_Ti_2* (2*X+Y-1)**2-Z*(2*L_Al_Ti_2*(2*X+Y-1)+L_Al_Ti_1)-X*(-4*L_Al_Ti_2*(2*X+Y-1)-\
            2*L_Al_Ti_1)+L_Al_Ti_1*(2*X+Y-1)-4 *L_Al_Ti_2*Z*X+(L_Al_Ti_1-2*L_Al_Ti_2*(-2*X-Y+1))*X+L_Al_Ti_0
    term3 = L_Nb_Ti_0
    DDex1_32 = term1 + term2 + term3
    DDex2_32 = (L_Al_Nb_Ti_1-L_Al_Nb_Ti_2)*(X*Y-Z*Y)+(2*Y+2*X-1)*(L_Al_Nb_Ti_1*Y+L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X)-(L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*X*Y+(L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*X*Z
    return (DDid_32 + DDex1_32 + DDex2_32)/VM
@numba.jit
def DDgibbsliq_32(X, Y, T, R, VM):
    Z = 1-X-Y
    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)
    DDid_32 = - R * T / Z
    term1 = -L_Al_Nb_2 * X ** 2 + X*(-2 * L_Al_Nb_2 * X - L_Al_Nb_1) - L_Al_Nb_1 * X - L_Al_Nb_0
    term2 = L_Al_Ti_2*(2 * X + Y - 1) ** 2 - Z * (2 * L_Al_Ti_2 * (2 * X + Y - 1) + L_Al_Ti_1) - X * (
                -4 * L_Al_Ti_2 * (2 * X + Y - 1) - \
                2 * L_Al_Ti_1) + L_Al_Ti_1 * (2 * X + Y - 1) - 4 * L_Al_Ti_2 * Z * X + (
                        L_Al_Ti_1 - 2 * L_Al_Ti_2 * (-2 * X - Y + 1)) * X + L_Al_Ti_0
    term3 = L_Nb_Ti_0
    DDex1_32 = term1 + term2 + term3
    DDex2_32 = (L_Al_Nb_Ti_1 - L_Al_Nb_Ti_2)*(X * Y - Z * Y) + (2 * Y + 2 * X - 1)*(
        L_Al_Nb_Ti_1 * Y + L_Al_Nb_Ti_2 * Z + L_Al_Nb_Ti_0 * X) - (L_Al_Nb_Ti_0 - L_Al_Nb_Ti_2) * X * Y + (
                           L_Al_Nb_Ti_0 - L_Al_Nb_Ti_2) * X * Z
    return (DDid_32 + DDex1_32 + DDex2_32) / VM
@numba.jit
def DDgibbsBCC_33(X, Y, T, R, VM):
    Z = 1-X-Y
    L_Al_Nb_0 = BCC_L_Al_Nb_0(T)
    L_Al_Nb_1 = BCC_L_Al_Nb_1(T)
    L_Al_Nb_2 = BCC_L_Al_Nb_2(T)

    L_Al_Ti_0 = BCC_L_Al_Ti_0(T)
    L_Al_Ti_1 = BCC_L_Al_Ti_1(T)
    L_Al_Ti_2 = BCC_L_Al_Ti_2(T)

    L_Nb_Ti_0 = BCC_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = BCC_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = BCC_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = BCC_L_Al_Nb_Ti_2(T)
    DDid_33 =  R*T *(1/Z+1/X)

    term1 = 2 * L_Al_Nb_2 * Y*(-Z - 2 * Y + 1) - 2 * Y * (-2 * L_Al_Nb_2 * (-Z - 2 * Y + 1) - L_Al_Nb_1)
    term2 = (-2 * Y * (L_Al_Nb_Ti_2 * Z + L_Al_Nb_Ti_0 * X + L_Al_Nb_Ti_1 * Y) - \
             2 * (L_Al_Nb_Ti_2 - L_Al_Nb_Ti_0) * Y * Z + 2 * (L_Al_Nb_Ti_2 - L_Al_Nb_Ti_0) * Y * X)
    term3 = 0
    DDex1_33 = term1 + term2 + term3
    DDex2_33 = (L_Al_Nb_Ti_2 - L_Al_Nb_Ti_0)*(Y * X - Y * Z) - \
               2 * Y * (L_Al_Nb_Ti_2 * Z + L_Al_Nb_Ti_0 * X + L_Al_Nb_Ti_1 * Y) - (L_Al_Nb_Ti_0 - L_Al_Nb_Ti_2) * Y * Z +\
               (L_Al_Nb_Ti_0 - L_Al_Nb_Ti_2) * Y * X

    return (DDid_33 + DDex1_33 + DDex2_33) / VM
@numba.jit
def DDgibbsliq_33(X, Y, T, R, VM):
    Z = 1-X-Y
    L_Al_Nb_0 = liq_L_Al_Nb_0(T)
    L_Al_Nb_1 = liq_L_Al_Nb_1(T)
    L_Al_Nb_2 = liq_L_Al_Nb_2(T)

    L_Al_Ti_0 = liq_L_Al_Ti_0(T)
    L_Al_Ti_1 = liq_L_Al_Ti_1(T)
    L_Al_Ti_2 = liq_L_Al_Ti_2(T)

    L_Nb_Ti_0 = liq_L_Nb_Ti_0(T)
    L_Al_Nb_Ti_0 = liq_L_Al_Nb_Ti_0(T)
    L_Al_Nb_Ti_1 = liq_L_Al_Nb_Ti_1(T)
    L_Al_Nb_Ti_2 = liq_L_Al_Nb_Ti_2(T)
    DDid_33 = R*T*(1/Z+1/X)

    term1 = 2*L_Al_Nb_2*Y*(-Z-2*Y+1)-2*Y*(-2*L_Al_Nb_2*(-Z-2*Y+1)-L_Al_Nb_1)
    term2 = -2* Y*(L_Al_Nb_Ti_2*Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)-2*(L_Al_Nb_Ti_2-L_Al_Nb_Ti_0)*Y*Z+2*(L_Al_Nb_Ti_2-L_Al_Nb_Ti_0)*Y*X
    term3 = 0
    DDex1_33 = term1 + term2+term3
    DDex2_33 =(L_Al_Nb_Ti_2-L_Al_Nb_Ti_0)*(Y*X- Y*Z)-2* Y* (L_Al_Nb_Ti_2* Z+L_Al_Nb_Ti_0*X+L_Al_Nb_Ti_1*Y)-(L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*Y*Z+(L_Al_Nb_Ti_0-L_Al_Nb_Ti_2)*Y*X

    return (DDid_33 + DDex1_33 + DDex2_33) / VM
@numba.jit
def mu1_s_tilda(X, Y, T, R, VM):
    Z = 1-X-Y
    return (gibbsBCC_Al(T) - gibbsBCC_Ti(T) + R *T *(safe_log(X) - safe_log(Z)))/VM
@numba.jit
def mu1_l_tilda(X, Y,T, R, VM):
    Z = 1 - X - Y
    return (gibbsliq_Al(T) - gibbsliq_Ti(T) + R * T * (safe_log(X) - safe_log(Z))) / VM

@numba.jit
def mu2_s_tilda(X, Y, T, R, VM):
    Z = 1 - X - Y
    return (gibbsBCC_Nb(T) - gibbsBCC_Ti(T) + R * T * (math.log(Y) - safe_log(Z))) / VM

@numba.jit
def mu2_l_tilda(X, Y, T, R, VM):
    Z = 1 - X - Y
    return (gibbsliq_Nb(T) - gibbsliq_Ti(T) + R * T * (safe_log(Y) - safe_log(Z))) / VM
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
    beta_out = fields_out[16]
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
    Xoffs = params[22]
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

            N = 2 # (N phase)

            #(37) mu_n interface mob (aniso)
            SUM_ = (c1_l[i][j] - c1_s[i][j])**2/P1 + (c2_l[i][j] - c2_s[i][j])**2/P2
            K = 4 * N * xi * mu_n / (4 * N * xi + mu_n * math.pi**2 * SUM_)

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
            dphidt = K/N * (sigma_n * (I_s - I_l) + (math.pi ** 2) * dG / (4 * xi))
            curr_phi = phi[i][j]

            eta = 0.01
            betaval = beta[i][j] - 0.5
          
            #phi_out[i,j] = curr_phi + dt*dphidt + eta*betaval*math.sqrt(dt)
            phi_out[i, j] = curr_phi + dt * dphidt

            if phi_out[i][j] < 0.0:
                phi_out[i][j] = 0
            elif phi_out[i][j] > 1.0:
                phi_out[i][j] = 1
            
            beta_out[i][j] = beta[i][j]

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
def solveD(fields,params,fields_out):
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

    D_11_s_out = fields_out[8]
    D_12_s_out = fields_out[9]
    D_21_s_out = fields_out[10]
    D_22_s_out = fields_out[11]
    D_11_l_out = fields_out[12]
    D_12_l_out = fields_out[13]
    D_21_l_out = fields_out[14]
    D_22_l_out = fields_out[15]

    R = params[10]
    VM = params[11]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):

            T = Tarr[i][j]
            c3_l = 1 - c1_l[i][j] - c2_l[i][j]
            c3_s = 1 - c1_s[i][j] - c2_s[i][j]

            #mob of Al
            phi_al_al = - 215000-80.2 *T #[4]
            phi_nb_al = - 187609.1 - 369.5*T #[4]
            term1 = 5.51e-6 * math.exp(-204000 / (T * R))
            term2 = 5.19e-10 * math.exp(-96000 / (T * R))
            phi_ti_al = R * T * math.log(term1 + term2) #[4]

            phi0_alnb_al = - 103573.3-318.2*T #[4]
            phi1_alnb_al = -1893639 # [4]
            phi0_alti_al = - 146284.8+65.7*T #[4]
            phi1_alti_al = 0
            phi0_nbti_al = 325981+100.5*T #[4]
            phi1_nbti_al = 0
            # mob of Nb
            phi_nb_nb = -395598.95 - 82.03 * T  # [4]
            phi_al_nb = -102306.3 - 122.4 *T # [4]
            term1 = 7.48e-2 * math.exp(-354800 / (T * R))
            term2 = 4.06e-8 * math.exp(-135800 / (T * R))
            phi_ti_nb =R * T * math.log(term1 + term2) #[4] # phi_ti_nb = -171237.75 - 115.83 * T [3]

            phi0_alnb_nb = - 101278.6 - 92.07*T
            phi0_nbti_nb = 175580-82.4*T # = 107764.17 - 14.52 * T [3]
            phi0_alti_nb =  - 99153.3+28.04*T
            phi0_alnbti_nb = - 81715.7-67.06*T

            # mob of Ti
            phi_al_ti = -366156.2 + 48.7 * T  # [4]
            phi_nb_ti =  -372297- 84.1*T # -369002.77 - 87.15 * T #[3]
            term1 = 5.91e-5 * math.exp(-237000 / (T * R))
            term2 = 1.47e-8 * math.exp(-121000 / (T * R))
            phi_ti_ti = R * T * math.log(term1 + term2) # phi_ti_ti = -151989.95 - 127.37 * T #[3]

            phi0_alti_ti = 59769.2-51.0*T
            phi0_nbti_ti = 106851-15.7*T #[4]  86711.4 + 2.61 * T
            phi0_alnb_ti = 522519

            # Redlick-kister polynomial
            term0_phi_Al_s= c1_s[i][j] * phi_al_al + c2_s[i][j] * phi_nb_al + c3_s * phi_ti_al
            term1_phi_Al_s = c1_s[i][j] * c2_s[i][j] *(phi0_alnb_al + phi1_alnb_al*(c1_s[i][j] - c2_s[i][j]) )
            term2_phi_Al_s = c1_s[i][j] * c3_s * (phi0_alti_al + phi1_alti_al * (c1_s[i][j] - c3_s))
            term3_phi_Al_s = c2_s[i][j] * c3_s * (phi0_nbti_al +phi1_nbti_al * (c2_s[i][j] - c3_s) )
            phi_Al_s = term0_phi_Al_s + term1_phi_Al_s + term2_phi_Al_s + term3_phi_Al_s

            term0_phi_Nb_s = c1_s[i][j] * phi_al_nb + c2_s[i][j] * phi_nb_nb + c3_s * phi_ti_nb
            term1_phi_Nb_s = c1_s[i][j] * c2_s[i][j] * phi0_alnb_nb
            term2_phi_Nb_s = c1_s[i][j] * c3_s * phi0_alti_nb
            term3_phi_Nb_s = c2_s[i][j] * c3_s * phi0_nbti_nb
            phi_Nb_s = term0_phi_Nb_s + term1_phi_Nb_s + term2_phi_Nb_s + term3_phi_Nb_s

            term0_phi_Ti_s = c1_s[i][j] * phi_al_ti + c2_s[i][j] * phi_nb_ti + c3_s * phi_ti_ti
            term1_phi_Ti_s = c1_s[i][j] * c2_s[i][j] * phi0_alnb_ti
            term2_phi_Ti_s = c1_s[i][j] * c3_s * phi0_alti_ti
            term3_phi_Ti_s = c2_s[i][j] * c3_s * phi0_nbti_ti
            phi_Ti_s = term0_phi_Ti_s+ term1_phi_Ti_s+ term2_phi_Ti_s+ term3_phi_Ti_s

            M_1_s = 1 / R / T * math.exp(phi_Al_s / R / T)
            M_2_s = 1 / R / T * math.exp(phi_Nb_s / R / T)
            M_3_s = 1 / R / T * math.exp(phi_Ti_s / R / T)


            Dmiu1_c1 = DDgibbsBCC_11(c1_s[i][j], c2_s[i][j], T, R, VM)
            Dmiu2_c2 = DDgibbsBCC_22(c1_s[i][j], c2_s[i][j], T, R, VM)
            Dmiu3_c3 = DDgibbsBCC_33(c1_s[i][j], c2_s[i][j], T, R, VM)

            Dmiu1_c2 = DDgibbsBCC_12(c1_s[i][j], c2_s[i][j], T, R, VM)
            Dmiu2_c1 = Dmiu1_c2  # DDgibbsBCC_21(c1_s[i][j], c2_s[i][j], T, R, VM)
            Dmiu3_c1 = DDgibbsBCC_31(c1_s[i][j], c2_s[i][j], T, R, VM)
            Dmiu1_c3 = Dmiu3_c1 # DDgibbsBCC_13(c1_s[i][j], c2_s[i][j], T, R, VM)
            Dmiu3_c2 = DDgibbsBCC_32(c1_s[i][j], c2_s[i][j], T, R, VM)
            Dmiu2_c3 = Dmiu3_c2# DDgibbsBCC_23(c1_s[i][j], c2_s[i][j], T, R, VM)


            # eq A.1 [1]
            D_11_s_out[i][j] = (1 - c1_s[i][j]) * c1_s[i][j] * M_1_s * (Dmiu1_c1 - Dmiu1_c3) + \
                               (-c1_s[i][j]) * c2_s[i][j] * M_2_s * (Dmiu2_c1 - Dmiu2_c3) + \
                               (-c1_s[i][j]) * c3_s * M_3_s * (Dmiu3_c1 - Dmiu3_c3)

            D_12_s_out[i][j] = (1 - c1_s[i][j]) * c1_s[i][j] * M_1_s * (Dmiu1_c2 - Dmiu1_c3) + \
                               (-c1_s[i][j]) * c2_s[i][j] * M_2_s * (Dmiu2_c2 - Dmiu2_c3) + \
                               (-c1_s[i][j]) * c3_s * M_3_s * (Dmiu3_c2 - Dmiu3_c3)

            D_21_s_out[i][j] = (-c2_s[i][j]) * c1_s[i][j] * M_1_s * (Dmiu1_c1 - Dmiu1_c3) + \
                               (1 - c2_s[i][j]) * c2_s[i][j] * M_2_s * (Dmiu2_c1 - Dmiu2_c3) + \
                               (-c2_s[i][j]) * c3_s * M_3_s * (Dmiu3_c1 - Dmiu3_c3)

            D_22_s_out[i][j] = (-c2_s[i][j]) * c1_s[i][j] * M_1_s * (Dmiu1_c2 - Dmiu1_c3) + \
                               (1 - c2_s[i][j]) * c2_s[i][j] * M_2_s * (Dmiu2_c2 - Dmiu2_c3) + \
                               (-c2_s[i][j]) * c3_s * M_3_s * (Dmiu3_c2 - Dmiu3_c3)

#----------------------------------------liquid diff----------------------------------------------
            # phi_al_al_l = 18912.7 - 138.68*T #[2]
            # mob of Al
            phi_al_al_l = -12137.49509 + math.log(5.28e-08) * T
            phi_nb_al_l = -12137.49509 + math.log(5.68283E-08) * T
            phi_ti_al_l = -12137.49509 + math.log(5.70616e-08) * T

            # mob of Nb
            phi_al_nb_l = -66079.56823 + math.log(5.51949e-08) * T
            phi_nb_nb_l = -66079.56823 + math.log(5.9406e-08) * T
            phi_ti_nb_l = -66079.56823+math.log(5.96498e-08) * T

            #mob of Ti
            phi_al_ti_l = -46640.1607 + math.log(6.78738e-08) * T
            phi_nb_ti_l = -46640.1607 +math.log(7.30522e-08) * T
            phi_ti_ti_l = -46640.1607 + math.log(7.3352e-08) * T

            phi_Al_l = c1_l[i][j] * phi_al_al_l + c2_l[i][j] * phi_nb_al_l + c3_l * phi_ti_al_l
            phi_Nb_l = c1_l[i][j] * phi_al_nb_l + c2_l[i][j] * phi_nb_nb_l + c3_l * phi_ti_nb_l
            phi_Ti_l = c1_l[i][j] * phi_al_ti_l + c2_l[i][j] * phi_nb_ti_l + c3_l * phi_ti_ti_l

            M_1_l = 1 / R / T * math.exp(phi_Al_l / R / T)
            M_2_l = 1 / R / T * math.exp(phi_Nb_l / R / T)
            M_3_l = 1 / R / T * math.exp(phi_Ti_l / R / T)

            Dmiu1_c1 = DDgibbsliq_11(c1_l[i][j], c2_l[i][j], T, R, VM)
            Dmiu1_c2 = DDgibbsliq_12(c1_l[i][j], c2_l[i][j], T, R, VM)

            Dmiu2_c1 = Dmiu1_c2 # DDgibbsliq_21(c1_l[i][j], c2_l[i][j], T, R, VM)
            Dmiu2_c2 = DDgibbsliq_22(c1_l[i][j], c2_l[i][j], T, R, VM)

            Dmiu3_c1 = DDgibbsliq_31(c1_l[i][j], c2_l[i][j], T, R, VM)
            Dmiu1_c3 = Dmiu3_c1 #DDgibbsliq_13(c1_l[i][j], c2_l[i][j], T, R, VM)
            Dmiu3_c2 = DDgibbsliq_32(c1_l[i][j], c2_l[i][j], T, R, VM)
            Dmiu2_c3 = Dmiu3_c2 #DDgibbsliq_23(c1_l[i][j], c2_l[i][j], T, R, VM)
            Dmiu3_c3 = DDgibbsliq_33(c1_l[i][j], c2_l[i][j], T, R, VM)

            D_11_l_out[i][j] = (1 -c1_l[i][j])*c1_l[i][j]*M_1_l*( Dmiu1_c1 - Dmiu1_c3) + \
                               (-c1_l[i][j])* c2_l[i][j]*M_2_l * (Dmiu2_c1-Dmiu2_c3 ) + \
                               (-c1_l[i][j]) * c3_l * M_3_l * (Dmiu3_c1 - Dmiu3_c3)

            D_12_l_out[i][j] = (1 -c1_l[i][j])*c1_l[i][j] * M_1_l * (Dmiu1_c2 - Dmiu1_c3) + \
                               (-c1_l[i][j]) * c2_l[i][j] * M_2_l * (Dmiu2_c2 - Dmiu2_c3) + \
                               (-c1_l[i][j]) * c3_l * M_3_l * (Dmiu3_c2 - Dmiu3_c3)

            D_21_l_out[i][j] = (-c2_l[i][j]) *c1_l[i][j] * M_1_l * (Dmiu1_c1 - Dmiu1_c3)+ \
                               (1-c2_l[i][j])*c2_l[i][j] * M_2_l * (Dmiu2_c1-Dmiu2_c3 ) + \
                               (-c2_l[i][j]) *c3_l * M_3_l * (Dmiu3_c1 -Dmiu3_c3)

            D_22_l_out[i][j] = (-c2_l[i][j])* c1_l[i][j] * M_1_l * (Dmiu1_c2 - Dmiu1_c3)+ \
                               (1-c2_l[i][j])* c2_l[i][j]* M_2_l * (Dmiu2_c2 - Dmiu2_c3) + \
                               (-c2_l[i][j]) * c3_l * M_3_l * (Dmiu3_c2 - Dmiu3_c3)

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

            mu1_s = mu1_s_tilda(c1_s[i][j], c2_s[i][j], T, R, VM)  #
            mu1_l = mu1_l_tilda(c1_l[i][j], c2_l[i][j], T, R, VM)  #
            mu2_s = mu2_s_tilda(c1_s[i][j], c2_s[i][j], T, R, VM)  #
            mu2_l = mu2_l_tilda(c1_l[i][j], c2_l[i][j], T, R, VM)  #

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

    Ds_21 = Ds_12
    Dl_21 = Dl_12

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

    Tarr_out = fields_out[7]

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
 
    Tarr_out = fields_out[7]
    xoffs = params[18]
    M = Tarr.shape[1]
    c1_0 = params[14]
    c2_0 = params[15]
    kini_1 = params[12]
    kini_2 = params[13]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1]- 1, stridex):

            if j < xoffs:
                phi_out[i][j] = phi_out[i][j + xoffs]
                c1_out[i][j] = c1_out[i][j+ xoffs]
                c1_l_out[i][j] = c1_l_out[i][j + xoffs]
                c1_s_out[i][j] = c1_s_out[i][j + xoffs]

            else:
                phi_out[i][j] = 0
               
                c1_s_out[i][j] = c1_0 * kini_1 #c_s_out[i][M-2]  # c_s[i, M-1]
                c1_l_out[i][j] = c1_0 #c_l_out[i][M-2]  # c_l[i, M-1]

            cuda.syncthreads()

class old_fid_eng(Simulation):
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
        D_11_l =  np.zeros(dim)
        D_12_l = np.zeros(dim)
        D_21_l = np.zeros(dim)
        D_22_l = np.zeros(dim)
        D_11_s =  np.zeros(dim)
        D_12_s = np.zeros(dim)
        D_21_s = np.zeros(dim)
        D_22_s = np.zeros(dim)
        np.random.seed(42)
        beta = np.random.rand(N, M).astype(np.float32) 

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

        self.user_data["params"] = np.array(params)
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])


    def simulation_loop(self):

        cuda.synchronize()

        solvePhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()
        #solveD[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        solveC_const_D[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        cuda.synchronize()

        #solveC_Dij[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        cuda.synchronize()

        TarrUpdate[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device,self.time_step_counter -1)
        cuda.synchronize()

        #thresh = int(0.5*3000)
        #if (self.time_step_counter - 1) % 500000:
            # phi = self._fields_out_gpu_device[0]
            # phi_column_j1000 = [phi[i][thresh] for i in range(len(phi))]
            #maxphi_thresh = max(phi_column_j1000)
            #if maxphi_thresh > 0.1:
            #self.user_data["params_GPU"][18] += self.dx * 200
            #PullBack[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

            #cuda.synchronize()

        if (self.time_step_counter -1) % 50000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    == 0:
            self.retrieve_fields_from_GPU()
            output_folder = self.user_data["save_path"]
            field_names = ['phi', 'c1', 'c1_s', 'c1_l', 'c2', 'c2_s', 'c2_l', 'tarr']
            os.makedirs(output_folder, exist_ok=True)
            for field, name in zip(self.fields, field_names):
                # Convert the field data to a DataFrame
                df = pd.DataFrame(field.data)
                # Construct the filename using the field name and the current time step counter
                filename = os.path.join(output_folder, f'{name}_step_{self.time_step_counter - 1}.csv')
                # Save the DataFrame to CSV without an index and without a header
                df.to_csv(filename, index=False, header=False)



        








