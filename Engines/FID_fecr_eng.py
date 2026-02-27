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
# pay attention to the BC init and the output when data transtfer from GPU, the random for interface has been set
#此处使用了原来jose的代码，但是能量是来自文章，Nb-Ti的，L依旧是简单常数版本
@cuda.jit(device=True)
def safe_log(x):
    if x <= 0.0:
        return 0.0  # 或 return -1e20，如果你想让它返回极小值
    return math.log(x)
@cuda.jit(device=True) #?

def gibbsliq_Ti(T): #?
    return -19887.006 + 298.7367 * T - 46.29 * T * safe_log(T)
    #return 369519.198 - 2554.0225*T + 342.059267 * T * safe_log(T) - 163.409355E-3 *T**2+12.457117E-6*T**3 - 67034516/T

@cuda.jit(device=True) #?
def gibbsliq_Nb(T):
    return 21262.202 + 131.229057*T - 26.4711*T * math.log(T)+0.203475E-3*T**2 - 0.350119E-6*T**3 + 93399/T - 3.06098E-23*T**7
@cuda.jit(device=True) #?

def gibbsBCC_Ti(T):
    #return 5511.037 + 66.976538*T - 14.9466 * T * safe_log(T) - 8.1465E-3 * T**2 + 0.202715E-6 * T**3 -1477660/T
    return 6667.385+105.366379*T-22.3771*T*safe_log(T)+.00121707*T**2-8.4534E-07*T**3 -2002750/T
    
    #return -118526.786+638.706871*T - 87.2182461*T*safe_log(T)+ 8.204849E-3*T**2-0.304747e-6*T**3+36699805/T
    
@cuda.jit(device=True) #?

def gibbsBCC_Nb(T):
    #return -8519.353 + 142.045474 * T - 26.4711 * T * math.log(T)
    return  -37669.3 + 271.720843 * T - 41.77 * T * safe_log(T) + 1.52824E23*T**(-9)

@cuda.jit(device=True) #?

def gibbsliq(X, T, R, VM):
   
    gibbs_liq = R * T * (X * math.log(X) + (1.0 - X) * math.log(1.0 - X)) + (1-X)*gibbsliq_Ti(T) + X*gibbsliq_Nb(T) + X*(1-X)*7406.1
    return gibbs_liq/VM


@cuda.jit(device=True) #?
def gibbsFCC(X,T,R,VM,ke,me,Tm):
   
    gibbs_FCC = R * T * (X * math.log(X) + (1.0 - X) * math.log(1.0 - X)) + (1-X)*gibbsBCC_Ti(T) + X*gibbsBCC_Nb(T) + X*(1-X)*13045

    return gibbs_FCC/VM

@cuda.jit(device=True) #?
def Dgibbsliq(X,T,R,VM):
    
    D_gibbs_liq = R * T * (math.log(X) - math.log(1.0 - X))+ gibbsliq_Nb(T)- gibbsliq_Ti(T) +7406*(X-2*X)
    return D_gibbs_liq/VM

@numba.jit
def DgibbsFCC(X,T,R,VM,ke,me,Tm):
    result = R * T * (math.log(X) - math.log(1.0 - X)) + gibbsBCC_Nb(T) - gibbsBCC_Ti(T) +13045*(X-2*X)

    return result/VM


@cuda.jit
def solvePhi(fields,params,fields_out):
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    Tarr = fields[4]
    beta = fields[5]

    phi_out = fields_out[0]
    c_out = fields_out[1]

    dx = params[0]
    dt = params[1]
    P = params[2]
    mu = params[3]
    sigma = params[8]

    R = params[9]
    VM = params[10]
    ke = params[11]  # kini
    c_0 = params[12]
    me = params[13]
    k_an = params[14]
    v_an = params[15]
    Tm = params[16]
    Ts = params[17]
    Ds = params[20]
    Dl = params[19]
    xi = params[21]
 

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1,phi.shape[0] - 1,stridey):
        for j in range(startx + 1,phi.shape[1] -1,stridex):

            T = Tarr[i][j]
            f_S = gibbsFCC(c_s[i][j], T, R, VM,ke, me, Tm)
            f_L = gibbsliq(c_l[i][j], T, R, VM)
            mu_S = DgibbsFCC(c_s[i][j], T, R, VM,ke, me, Tm)
            mu_L = Dgibbsliq(c_l[i][j], T, R, VM)
            phi_l=1-phi[i][j]

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
            mu_n = mu * (1 - v_an * (3 - 4 * (normx**4 + normy**4))) #??
           
            #Calculate K (kinetic coefficient) and dG (driving force)
            K = 8 * P * xi * mu_n / (8 * P * xi + mu_n * math.pi**2 * (c_s[i][j] - c_l[i][j])**2)

            dG = f_L - f_S - (phi[i][j]*mu_S + (1-phi[i][j])*mu_L)*(c_l[i][j]-c_s[i][j])
            #dG=f_L-f_S - ((phi[i][j]*mu_S - phi_l*mu_L)*(c_s[i][j]-c_l[i][j]))
            
            
            laplacian_phi = (0.5 * (phi[i][j-1] + phi[i][j+1] + phi[i+1][j] + phi[i-1][j] + \
                                                0.5 * (phi[i+1][j+1] + phi[i-1][j+1] + phi[i-1][j-1] + \
                                                    phi[i+1][j-1]) - 6 * phi[i][j])) / (dx * dx)
            

            dphidt = K * (sigma_n * (laplacian_phi + math.pi**2 / xi**2 * (phi[i][j] - 0.5)) + math.pi/xi * math.sqrt(phi[i][j] * (1-phi[i][j])) * dG)
            curr_phi = phi[i][j]

            eta = 0.01
            betaval = beta[i][j] - 0.5
          
            phi_out[i,j] = curr_phi + dt*dphidt + eta*betaval*math.sqrt(dt)


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
def solveC(fields,params,fields_out):
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    Tarr = fields[4]
 
    phi_out = fields_out[0]  #
    c_out = fields_out[1]
    c_s_out = fields_out[2]
    c_l_out = fields_out[3]
   
    dx = params[0]
    dt = params[1]
    P = params[2]
    mu = params[3]
    sigma = params[8]

    R = params[9]
    VM = params[10]
    ke = params[11]  #kini
    c_0 = params[12]
    me = params[13]
    k_an = params[14]
    v_an = params[15]
    Tm = params[16]
  
    Ds = params[20]
    Dl = params[19]
    xi = params[21]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):

            T = Tarr[i][j]
            c_l_out[i][j] = 0.5
            #phi next has been calculated in solv_phi
            dphidt_ij = (phi_out[i][j] - phi[i][j]) / dt
            dphidt_opp = ((1-phi_out[i,j]) - (1-phi[i,j])) / dt

            mu_S = DgibbsFCC(c_s[i][j], T,R,VM,ke,me,Tm)
            mu_L = Dgibbsliq(c_l[i][j], T, R, VM)

            xplus_s = ((phi_out[i][j] * Ds + phi_out[i+1][j] * Ds) / 2) * ((c_s[i+1][j] - c_s[i][j]) / dx)
            xmins_s = ((phi_out[i][j] * Ds + phi_out[i-1][j] * Ds) / 2) * ((c_s[i-1][j] - c_s[i][j]) / dx)
            yplus_s = ((phi_out[i][j] * Ds + phi_out[i][j+1] * Ds) / 2) * ((c_s[i][j+1] - c_s[i][j]) / dx)
            ymins_s = ((phi_out[i][j] * Ds + phi_out[i][j-1] * Ds) / 2) * ((c_s[i][j-1] - c_s[i][j]) / dx)


            #liquid component of diffusion
            xplus_l = ((((1-phi_out[i][j]) * Dl + (1-phi_out[i+1][j]) * Dl) / 2) * (c_l[i+1][j] - c_l[i][j]) / dx)
            xmins_l = ((((1-phi_out[i][j]) * Dl + (1-phi_out[i-1][j]) * Dl) / 2) * (c_l[i-1][j] - c_l[i][j]) / dx)
            yplus_l = ((((1-phi_out[i][j]) * Dl + (1-phi_out[i][j+1]) * Dl) / 2) * (c_l[i][j+1] - c_l[i][j]) / dx)
            ymins_l = ((((1-phi_out[i][j]) * Dl + (1-phi_out[i][j-1]) * Dl) / 2) * (c_l[i][j-1] - c_l[i][j]) / dx)
            
            #dc/dt completed term
            if phi_out[i][j] < 1e-3: #if we are in liquid portion
                dcsdt = 0.0
                dcldt = (xplus_l + xmins_l + yplus_l + ymins_l) / dx
            elif phi_out[i][j] > 1-1e-3: #if we are in solid portion
                dcsdt = (xplus_s + xmins_s + yplus_s + ymins_s) / dx
                dcldt = 0.0
            else: #if(phi_out[i,j] >= 1e-9 && phi_out[i,j] <= 1-1e-9) #if we are at the interface solve both equations
                dcsdt = ((xplus_s + xmins_s + yplus_s + ymins_s) / dx + P*phi_out[i][j]*(1-phi_out[i][j])*(mu_L - mu_S) + phi_out[i][j] * dphidt_ij * (c_l[i][j] - c_s[i][j])) / phi_out[i][j]
                dcldt = ((xplus_l + xmins_l + yplus_l + ymins_l) / dx + P*phi_out[i][j]*(1-phi_out[i][j])*(mu_S - mu_L) + (1-phi_out[i][j]) * dphidt_opp * (c_s[i][j] - c_l[i,j])) / (1-phi_out[i][j])


            c_s_out[i][j] = c_s[i][j] + dt * dcsdt
            c_l_out[i][j] = c_l[i][j] + dt * dcldt
            c_out[i][j] = c_s_out[i][j] * phi_out[i][j] + c_l_out[i][j] * (1- phi_out[i][j])

            if c_s_out[i][j] < 0.0:
                c_s_out[i][j] = 0.0000001
            elif c_s_out[i][j] > 1.0:
                c_s_out[i][j] = 0.9999999
            
            if c_l_out[i][j] < 0.0:
                c_l_out[i][j] = 0.0000001
            elif c_l_out[i][j] > 1.0:
                c_l_out[i][j] = 0.9999999
          
#BC
@cuda.jit
def TarrUpdate(fields,params,fields_out,timestep):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    Tarr = fields[4]
    beta = fields[5]

    Tarr_out = fields_out[4]
    beta_out = fields_out[5]

    dx = params[0]
    dt = params[1]
    Vs = params[4]
    G = params[5]
    initXpos = params[7]
    Ts = params[17]
    xoffs = params[18]

    t = timestep * dt
    for i in range(starty, Tarr.shape[0], stridey):
        for j in range(startx, Tarr.shape[1], stridex):

            #Tarr_out[i][j] = (T0 + G * ((j-1) * dx + xoffs - Vs * t))
            Tarr_out[i][j] = (Ts + G * ((j-1) * dx - initXpos * dx + xoffs - Vs * t))

            beta_out[i][j] = beta[i][j]
@cuda.jit
def initBC(fields):
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty,phi.shape[0],stridey):
        for j in range(startx,phi.shape[1],stridex):
            ##### for ic initiate
            if i == 0 or i == phi.shape[0]:
                for j in range(phi.shape[1]):
                    phi[0][j] = phi[1][j]
                    phi[phi.shape[0]][j] = phi[phi.shape[0] - 1][j]
                    c[0][j] = c[1][j]
                    c[phi.shape[0]][j] = c[phi.shape[0] - 1][j]
                    c_l[0][j] = c_l[1][j]
                    c_l[phi.shape[0]][j] = c_l[phi.shape[0] - 1][j]
                    c_s[0][j] = c_s[1][j]
                    c_s[phi.shape[0]][j] = c_s[phi.shape[0] - 1][j]
                   

            if j == 0 or j == phi.shape[1]:
                for i in range(phi.shape[0]):
                    phi[i][0] = phi[i][1]
                    phi[i][phi.shape[1]] = phi[i][phi.shape[1] - 1]
                    c[i][0] = c[i][1]
                    c[i][phi.shape[1]] = c[i][phi.shape[1] - 1]
                    c_s[i][0] = c_s[i][1]
                    c_s[i][phi.shape[1]] = c_s[i][phi.shape[1] - 1]
                    c_l[i][0] = c_l[i][1]
                    c_l[i][phi.shape[1]] = c_l[i][phi.shape[1] - 1]
                   

@cuda.jit
def PullBack(fields,params,fields_out):

    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
  
    Tarr = fields[4]

    phi_out = fields_out[0]
    c_out = fields_out[1]
    c_s_out = fields_out[2]
    c_l_out = fields_out[3]
 
    Tarr_out = fields_out[4]
    xoffs = params[18]
    M = Tarr.shape[1]
    c_0 = params[15]
    kini = params[14]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1]- 1, stridex):

            if j < xoffs:
                phi_out[i][j] = phi_out[i][j + xoffs]
                c_out[i][j] = c_out[i][j+ xoffs]
                c_l_out[i][j] = c_l_out[i][j + xoffs]
                c_s_out[i][j] = c_s_out[i][j + xoffs]


            else:
                phi_out[i][j] = 0
               
                c_s_out[i][j] = c_0 * kini #c_s_out[i][M-2]  # c_s[i, M-1]
                c_l_out[i][j] = c_0 #c_l_out[i][M-2]  # c_l[i, M-1]
                

            cuda.syncthreads()

class fid_fecr_eng(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL"

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        N = dim[0] #2000
        M = dim[1] #3000
        phi = np.zeros(dim)
        c = np.zeros(dim)
        c_s = np.ones(dim)
        c_l = np.ones(dim)

        Tarr = np.zeros(dim)

        beta = np.random.rand(N,M).astype(np.float32) #
        Ds = np.zeros(dim)
        Dl = np.zeros(dim)
        dcdt = np.zeros(dim)

        xi = self.user_data["xi"]
        dx = self.dx
        initXpos = self.user_data["initXpos"]
        Ts = self.user_data["Ts"]
        G = self.user_data["G"]
        kini = self.user_data["kini"]

        c_0 = self.user_data["c_0"]
        # init Tarr
        # 创建 Tarr 的二维数组
        Tarr = Ts + G * (np.arange(M) * dx - initXpos * dx)  # 生成一维数组
        Tarr = np.tile(Tarr, (N, 1))  # 将 Tarr 一维数组复制成 N 行的二维数组

        #init phi
        posArr_h = initXpos * dx + (np.random.rand(N) - 0.5) * dx # 1D array
        posArr = np.zeros(N)
        posArr[:] = posArr_h

        x_p = np.arange(M) * dx  # 生成所有的 x_p
        phi = (1.0 - np.

               tanh((x_p - posArr[:, None]) / xi)) / 2  # 矢量化计算 phi

        phi = np.minimum(phi, 1.0)

        #init c_s and c_l
        c_s = c_s * c_0 * kini
        c_l = c_l * c_0

        self.add_field(phi, "phi",colormap=COLORMAP_PHASE)
        self.add_field(c, "c")
        self.add_field(c_s,"c_s")
        self.add_field(c_l, "c_l")
        self.add_field(Tarr,"Tarr")
        self.add_field(beta,"beta")

    def just_before_simulating(self):  #in simulation file
        super().just_before_simulating()

        params = []
        params.append(self.dx)
        params.append(self.dt)
        params.append(self.user_data["P"]) #2
        params.append(self.user_data["mu"]) #3
        params.append(self.user_data["Vs"]) #4
        params.append(self.user_data["G"]) #5
        params.append(self.user_data["dT"]) #6
        params.append(self.user_data["initXpos"]) #7
        params.append(self.user_data["sigma"]) #8
        
        params.append(self.user_data["R"]) #9
        params.append(self.user_data["VM"]) #10
        params.append(self.user_data["kini"])#11
        params.append(self.user_data["c_0"])#12
        params.append(self.user_data["me"])#13
        params.append(self.user_data["k_an"])#14
        params.append(self.user_data["v_an"])#15
        params.append(self.user_data["Tm"])#16
        params.append(self.user_data["Ts"])#17
        
        params.append(self.user_data["xoffs"])#18
        
        params.append(self.user_data["Dl"])#19
        params.append(self.user_data["Ds"])#20

        params.append(self.user_data["xi"])#21


        self.user_data["params"] = np.array(params)
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])


    def simulation_loop(self):

        cuda.synchronize()

        solvePhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()
        solveC[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

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


        if (self.time_step_counter -1) % 500000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        == 0:
            self.retrieve_fields_from_GPU()
            output_folder = self.user_data["save_path"]
            os.makedirs(output_folder, exist_ok=True)
            phi = self.fields[0].data
            df = pd.DataFrame(phi)
            filename = os.path.join(output_folder, f'phi_step_{self.time_step_counter -1}.csv')
            df.to_csv(filename, index=False, header=False)

            c_s = self.fields[2].data
            df = pd.DataFrame(c_s)
            filename = os.path.join(output_folder, f'c_s_step{self.time_step_counter -1 }.csv')
            df.to_csv(filename, index=False, header=False)

            c = self.fields[1].data
            df = pd.DataFrame(c)
            filename = os.path.join(output_folder, f'c_step{self.time_step_counter - 1}.csv')
            df.to_csv(filename, index=False, header=False)

            c_l = self.fields[3].data
            df = pd.DataFrame(c_l)
            filename = os.path.join(output_folder, f'c_l_step{self.time_step_counter - 1}.csv')
            df.to_csv(filename, index=False, header=False)

            tarr = self.fields[4].data
            df = pd.DataFrame(tarr)
            filename = os.path.join(output_folder, f'tarr_step{self.time_step_counter -1 }.csv')
            df.to_csv(filename, index=False, header=False)


        








