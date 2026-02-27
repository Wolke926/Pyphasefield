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
#no moving frame


@cuda.jit
def PullBack(fields,params,fields_out):

    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    theta = fields[4]
    Tarr = fields[5]

    phi_out = fields_out[0]
    c_out = fields_out[1]
    c_s_out = fields_out[2]
    c_l_out = fields_out[3]
    theta_out = fields_out[4]
    Tarr_out = fields_out[5]

    M = Tarr.shape[1]
    c_0 = params[15]
    kini = params[14]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1]- 1, stridex):




            if j < 2000 :
                phi_out[i][j] = phi_out[i][j + 1]
                c_out[i][j] = c_out[i][j + 1]
                c_l_out[i][j] = c_l_out[i][j + 1]
                c_s_out[i][j] = c_s_out[i][j + 1]
                theta_out[i][j] = theta_out[i][j + 1]

            else:
                phi_out[i][j] = 0
                c_out[i][j] = 0.5 #c[i,j]
                c_s_out[i][j] = c[i,j] * kini / (kini * phi[i,j] - phi[i,j] + 1) #c_s_out[i][M-2]  # c_s[i, M-1]
                c_l_out[i][j] = c[i,j] / (kini * phi[i,j] - phi[i,j] + 1) #c_l_out[i][M-2]  # c_l[i, M-1]
                theta_out[i][j] = 0# theta_out[i][j-1]

            cuda.syncthreads()
@cuda.jit
def PullBack2(fields,params,fields_out):

    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    theta = fields[4]
    Tarr = fields[5]

    phi_out = fields_out[0]
    c_out = fields_out[1]
    c_s_out = fields_out[2]
    c_l_out = fields_out[3]
    theta_out = fields_out[4]
    Tarr_out = fields_out[5]

    M = Tarr.shape[1]
    c_0 = params[15]
    kini = params[14]
    i,j = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    phi_out[i][j] = phi[i][j]
    c_out[i][j] = c[i][j]
    c_l_out[i][j] = c_l[i][j]
    c_s_out[i][j] = c_s[i][j]
    theta_out[i][j] = theta[i][j]

    if j < M:
        phi_out[i][j] = phi_out[i][j + 1]
        c_out[i][j] = c_out[i][j + 1]
        c_l_out[i][j] = c_l_out[i][j + 1]
        c_s_out[i][j] = c_s_out[i][j + 1]
        theta_out[i][j] = theta_out[i][j + 1]


    else:
        phi_out[i][j] = 0
        c_out[i][j] = c[i,j]
        c_s_out[i][j] = c[i,j] * kini / (kini * phi[i,j] - phi[i,j] + 1) #c_s_out[i][M-2]  # c_s[i, M-1]
        c_l_out[i][j] = c[i,j] / (kini * phi[i,j] - phi[i,j] + 1) #c_l_out[i][M-2]  # c_l[i, M-1]
        theta_out[i][j] = theta_out[i][j-1]

class mf_test(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL"

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        N = dim[0]
        M = dim[1]
        phi = np.zeros(dim)
        c = np.full((dim),0.5)
        c_s = np.ones(dim)
        c_l = np.ones(dim)
        theta = np.zeros(dim)
        Tarr = np.zeros(dim)
        term1 = np.zeros(dim)
        term2 = np.zeros(dim)
        term3 = np.zeros(dim)
        allterm = np.zeros(dim)
        Ds = np.zeros(dim)
        Dl = np.zeros(dim)
        dcdt = np.zeros(dim)

        Tl= self.user_data["Tl"]
        xi = self.user_data["xi"]
        dx = self.dx
        initXpos = self.user_data["initXpos"]
        T0 = self.user_data["T0"]
        G = self.user_data["G"]
        kini = self.user_data["kini"]
        c_0 = self.user_data["c_0"]
        # init Tarr




        for i in range(N):
            for j in range(0,M//2):

                phi[i][j] = 1
                c[i][j] = 0.6 #
                theta[i][j] = 0.5 #
                Tarr[i][j] = 1000



        #init c


        #init c_s and c_l

        #init theta


        #注意这里的bc 似乎不需要


        self.add_field(phi, "phi",colormap=COLORMAP_PHASE)
        self.add_field(c, "c")
        self.add_field(c_s,"c_s")
        self.add_field(c_l, "c_l")
        self.add_field(theta, "theta")
        self.add_field(Tarr,"Tarr")
        self.add_field(term1, "term1")
        self.add_field(term2, "term2")
        self.add_field(term3, "term3")
        self.add_field(allterm, "allterm")
        self.add_field(Ds, "Ds")
        self.add_field(Dl, "Dl")
        self.add_field(dcdt, "dcdt")


    def just_before_simulating(self):  #in simulation file
        super().just_before_simulating()

        params = []
        params.append(self.dx)
        params.append(self.dt)
        params.append(self.user_data["r"])
        params.append(self.user_data["Vs"])
        params.append(self.user_data["G"])
        params.append(self.user_data["dT"])
        params.append(self.user_data["initXpos"])
        params.append(self.user_data["T0"])
        params.append(self.user_data["xi"])
        params.append(self.user_data["sigma"])
        params.append(self.user_data["epsilon"])
        params.append(self.user_data["w"])
        params.append(self.user_data["R"])
        params.append(self.user_data["VM"])
        params.append(self.user_data["kini"])
        params.append(self.user_data["c_0"])
        params.append(self.user_data["k_an"])
        params.append(self.user_data["v_an"])
        params.append(self.user_data["Mphi"])
        params.append(self.user_data["Tl"])

        self.user_data["params"] = np.array(params)
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])


    def simulation_loop(self):

        cuda.synchronize()

        PullBack[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        cuda.synchronize()

        if (self.time_step_counter) % 2 == 0:
            self.retrieve_fields_from_GPU()
            output_folder = 'mf_final'  ##
            os.makedirs(output_folder, exist_ok=True)
            phi = self.fields[0].data
            df = pd.DataFrame(phi)
            filename = os.path.join(output_folder, f'phi_step__{self.time_step_counter}.csv')
            df.to_csv(filename, index=False, header=False)

            c = self.fields[1].data
            df = pd.DataFrame(c)
            filename = os.path.join(output_folder, f'c_step{self.time_step_counter}.csv')
            df.to_csv(filename, index=False, header=False)









