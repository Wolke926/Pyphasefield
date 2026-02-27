#该版本进做功能测试使用 coupled with test_next.ipynb
import numpy as np
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
import pandas as pd
import os
try:
    from numba import cuda
    import numba
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
except:
    import pyphasefield.jit_placeholder as numba
    import pyphasefield.jit_placeholder as cuda

@cuda.jit
def solvePhi(fields,params,fields_out):

    phi = fields[0]

    phi_out = fields_out[0] #

    dd = params[1]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(startx,phi.shape[0],stridex):
        for j in range(starty,phi.shape[1],stridey):
            phi_out[i][j] = 0.678


@cuda.jit
def solveC(fields, params, fields_out):
    phi = fields[0]
    c = fields[2]
    phi_out = fields_out[0]  #
    c_out = fields_out[2]


    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(startx,phi.shape[0],stridex):
        for j in range(starty,phi.shape[1],stridey):
            c_out[i][j] = phi_out[i][j] - phi[i][j]


@cuda.jit
def minusPhi(fields,params,fields_out):

    phi = fields[0]
    phinext = fields[1]

    phi_out = fields_out[0] #
    phinext_out = fields_out[1]
    dx = params[0]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(startx,phi.shape[0],stridex):
        for j in range(starty,phi.shape[1],stridey):
            phinext_out[i][j] = - phi_out[i][j] + phi[i][j]

@cuda.jit
def UpdatePhi(fields,params,fields_out):

    phi = fields[0]
    phinext = fields[1]
    phi_out = fields_out[0]

    dx = params[0]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(startx,phi.shape[0],stridex):
        for j in range(starty,phi.shape[1],stridey):
            phinext[i][j] = phi_out[i][j] + 1

@cuda.jit
def test_phi_out(fields,params,fields_out,timestep):

    phi = fields[0]
    phi_out = fields_out[0]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(startx,phi.shape[0],stridex):
        for j in range(starty,phi.shape[1],stridey):
            phi_out[i][j] = 7 + timestep
@cuda.jit
def updateall(fields,params,fields_out,abc):
    c = fields[2]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(startx, c.shape[0], stridex):
        for j in range(starty, c.shape[1], stridey):
            c[i][j] = 0.111 # 0.111
            abc += c[i][j]



class PCR_test(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL"

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        phi = np.zeros(dim) # field0
        phinext = np.zeros(dim) #1
        c = np.ones(dim)


        #init dphidt/dcdt

        self.add_field(phi, "phi",colormap=COLORMAP_PHASE)
        self.add_field(phinext, "phinext")
        self.add_field(c,"c")

    def just_before_simulating(self):
        super().just_before_simulating()

        params = []
        params.append(self.dx)
        params.append(self.user_data["dd"])

        self.user_data["params"] = np.array(params)
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])


    def simulation_loop(self):
        cuda.synchronize()

        solvePhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        #minusPhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"], self._fields_out_gpu_device)
        #updateall[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device,abc)
        #cuda.synchronize()

        solveC[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        cuda.synchronize()
        # UpdatePhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
        #test_phi_out[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device,self.time_step_counter)



'''

        if self.time_step_counter % 10 == 0:
            output_folder = 'output'
            os.makedirs(output_folder, exist_ok=True)
            phi = self.fields[0].data
            df = pd.DataFrame(phi)
            filename = os.path.join(output_folder,f'{self.time_step_counter}TRY.csv')
            df.to_csv(filename, index=False, header=False)
'''