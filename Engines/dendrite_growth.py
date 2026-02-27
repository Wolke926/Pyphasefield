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
try:
    from numba import cuda
    import numba
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
except:
    import pyphasefield.jit_placeholder as numba
    import pyphasefield.jit_placeholder as cuda

@cuda.jit
def SolveT(fields,params,fields_out):
    phi = fields[0]
    T = fields[1]
    phi_out = fields_out[0]
    T_out = fields_out[1]
    dx = params[0]
    dt = params[1]
    D = params[7]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):
            lap_T = (T[i][j-1] + T[i][j+1] + T[i+1][j] + T[i-1][j] - 4 * T[i][j])/ (dx**2)
            dphidt = (phi_out[i][j] - phi[i][j]) /dt

            T_out[i][j] = T[i][j] + (D * lap_T + 0.5 * dphidt) * dt


@cuda.jit
def SolveDphidxy(fields,params,fields_out):
    phi = fields[0]
    phi_out = fields_out[0]
    dphidx_out = fields_out[3]
    dphidy_out = fields_out[4]

    dx = params[0]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):
            dphidx_out[i][j] = (phi_out[i+1][j] - phi_out[i-1][j])/(2*dx)
            dphidy_out[i][j] = (phi_out[i][j+1] - phi_out[i][j-1])/(2*dx)

@cuda.jit
def SolveTheta(fields,params,fields_out):
    phi = fields[0]
    dphidx_out = fields_out[3]
    dphidy_out = fields_out[4]
    theta_out = fields_out[2]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty, phi.shape[0], stridey):
        for j in range(startx, phi.shape[1], stridex):

            if (dphidx_out[i][j] ** 2 == 0)and (dphidy_out[i][j] ** 2 == 0):
                theta_out[i][j] = 0.0
            else:
                theta_out[i][j] = np.arctan2(dphidx_out[i][j], dphidy_out[i][j])


@cuda.jit
def SolvePhi(fields,params,fields_out):
    phi = fields[0]
    T = fields[1]
    theta = fields[2]
    dphidx = fields[3]
    dphidy = fields[4]
    dx = params[0]
    dt = params[1]
    W_0 = params[2]
    m = params[3]
    eta_4 = params[4]
    theta_0 = params[5]
    tau_0 = params[6]
    lamb = params[9]

    phi_out = fields_out[0]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):
            phi_curr = phi[i][j]
            term1 = (phi_curr - lamb * T[i][j] * (1 - phi_curr**2)) * (1 - phi_curr**2)  #

            a_n = 1 + eta_4 * math.cos(m * (theta[i][j] - theta_0))
            Wn = W_0 * a_n
            #lap_phi = (phi[i][j-1] + phi[i][j+1] + phi[i+1][j] + phi[i-1][j] - 4 * phi[i][j])
            #term2 = Wn**2 * lap_phi/ (dx**2) ##???

            W_n_ipos = 1 + eta_4 * math.cos(4 * (theta[i+1][j] - theta_0))
            W_n_ineg = 1 + eta_4 * math.cos(4 * (theta[i - 1][j] - theta_0))
            term2_x = ((dphidx[i+1][j] * W_n_ipos **2) - (dphidx[i-1][j] * W_n_ineg **2 ))/(2*dx)

            W_n_jpos = 1 + eta_4 * math.cos(4 * (theta[i][j + 1] - theta_0))
            W_n_jneg = 1 + eta_4 * math.cos(4 * (theta[i][j - 1] - theta_0))
            term2_y = ((dphidy[i][j+1] * W_n_jpos **2) - (dphidy[i][j-1] * W_n_jneg **2 ))/(2*dx)

            term2 = term2_x + term2_y #

            phi_x05_y05 = (phi[i][j] + phi[i+1][j] + phi[i+1][j+1] + phi[i][j+1])/4
            phi_x_05_y05 = (phi[i][j] + phi[i-1][j] + phi[i-1][j+1] + phi[i][j+1])/4
            phi_x_05_y_05 = (phi[i][j] + phi[i-1][j] + phi[i-1][j-1] + phi[i][j-1])/4
            phi_x05_y_05 = (phi[i][j] + phi[i][j-1] + phi[i+1][j-1] + phi[i+1][j])/4

            grad_phi_x05pos_sq = ((phi[i+1][j] - phi[i][j])/dx)**2 + ((phi_x05_y05 - phi_x05_y_05)/dx)**2 #

            theta_x05pos = (theta[i][j] + theta[i+1][j])/2
            a_n_x05pos = 1 + eta_4 * np.cos(theta_x05pos - theta_0)
            Wn_x05pos = W_0 * a_n_x05pos #



            grad_phi_x05neg_sq = ((phi[i][j] - phi[i-1][j] )/dx)**2 + ((phi_x_05_y05 - phi_x_05_y_05)/dx)**2 #

            theta_x05neg = (theta[i][j] + theta[i - 1][j]) / 2
            a_n_x05neg = 1 + eta_4 * np.cos(theta_x05neg - theta_0)
            Wn_x05neg = W_0 * a_n_x05neg #

            a_n_xpos = 1 + eta_4 * np.cos(theta[i+1][j] - theta_0)
            W_dphidx_pos = W_0 * a_n_xpos
            dW_d_dphidx_05pos = (W_dphidx_pos - Wn)/(dphidx[i+1][j] - dphidx[i][j])

            a_n_xneg = 1 + eta_4 * np.cos(theta[i-1][j] - theta_0)
            W_dphidx_neg = W_0 * a_n_xneg
            dW_d_dphidx_05neg = (Wn - W_dphidx_neg) / (dphidx[i][j] - dphidx[i-1][j])

            term3 = ((grad_phi_x05pos_sq * Wn_x05pos * dW_d_dphidx_05pos )- (grad_phi_x05neg_sq * Wn_x05neg * dW_d_dphidx_05neg) )/ dx #

            grad_phi_y05pos_sq = (( phi[i][j+1] - phi[i][j])/dx)**2 + ((phi_x05_y05 - phi_x_05_y05)/dx)**2 #
            grad_phi_y05neg_sq = (( phi[i][j] - phi[i][j-1])/dx)**2 + ((phi_x05_y_05 - phi_x_05_y_05)/dx)**2 #

            theta_y05pos = (theta[i][j] + theta[i][j+1]) / 2
            a_n_y05pos = 1 + eta_4 * np.cos(theta_y05pos - theta_0)
            Wn_y05pos = W_0 * a_n_y05pos  #

            theta_y05neg = (theta[i][j] + theta[i][j - 1]) / 2
            a_n_y05neg = 1 + eta_4 * np.cos(theta_y05neg - theta_0)
            Wn_y05neg = W_0 * a_n_y05neg  #

            a_n_ypos = 1 + eta_4 * np.cos(theta[i][j + 1] - theta_0)
            W_dphidy_pos = W_0 * a_n_ypos
            dW_d_dphidy_05pos = (W_dphidy_pos - Wn) / (dphidy[i][j + 1] - dphidy[i][j])

            a_n_yneg = 1 + eta_4 * np.cos(theta[i][j-1] - theta_0)
            W_dphidy_neg = W_0 * a_n_yneg
            dW_d_dphidy_05neg = (Wn - W_dphidy_neg) / (dphidy[i][j] - dphidy[i][j-1])

            term4 = ((grad_phi_y05pos_sq * Wn_y05pos * dW_d_dphidy_05pos ) - (grad_phi_y05neg_sq * Wn_y05neg * dW_d_dphidy_05neg) )/dx

            tau_n = tau_0 * a_n**2
            dphidt = (term1 + term2 + term3 + term4)/tau_n
            phi_out[i][j] = phi[i][j] + dphidt * dt
            if phi_out[i][j] > 1:
                phi_out[i][j] = 1.0
            if phi_out[i][j] < -1:
                phi_out[i][j] = -1


class dendrite_growth(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL"

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        phi = (-1) * np.ones(dim)
        T = self.user_data["delta"] * np.ones(dim)
        theta = np.zeros(dim)
        dphidx = np.zeros(dim)
        dphidy = np.zeros(dim)
        dx = self.dx

        N = dim[0]
        M = dim[1]
        for i in range(N):
            for j in range(M):
                if i**2 + j**2 <= 100:
                    phi[i][j] = 1.0

        # init theta
        for i in range(1, N - 1):
            for j in range(1, M - 1):

                dphidy[i][j] = (phi[i][j + 1] - phi[i][j - 1]) / (2 * dx)
                dphidx[i][j] = (phi[i + 1][j] - phi[i - 1][j]) / (2 * dx)
                if (dphidy[i][j] == 0.0) and (dphidx[i][j] == 0.0):
                    theta[i][j] = 0.0
                else:
                    theta[i][j] = np.arctan2(dphidy[i][j], dphidx[i][j])

        self.add_field(phi, "phi",colormap=COLORMAP_PHASE)
        self.add_field(T, "T")
        self.add_field(theta, "theta")
        self.add_field(dphidx, "dphidx")
        self.add_field(dphidy, "dphidy")

    def just_before_simulating(self):
        super().just_before_simulating()

        params = []
        params.append(self.dx)
        params.append(self.dt)
        params.append(self.user_data["W_0"])
        params.append(self.user_data["m"])
        params.append(self.user_data["eta_4"])
        params.append(self.user_data["theta_0"])
        params.append(self.user_data["tau_0"])
        params.append(self.user_data["D"])
        params.append(self.user_data["delta"])
        params.append(self.user_data["lamb"])

        self.user_data["params"] = np.array(params)
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])

    def simulation_loop(self):
        cuda.synchronize()

        SolvePhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()
        SolveT[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()

        SolveDphidxy[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,
                                                                                   self.user_data["params_GPU"],
                                                                                   self._fields_out_gpu_device)
        cuda.synchronize()
        SolveTheta[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,
                                                                                 self.user_data["params_GPU"],
                                                                                 self._fields_out_gpu_device)
        cuda.synchronize()




