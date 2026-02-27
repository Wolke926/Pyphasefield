import numpy as np

try:
        #import from within Engines folder
        from ..field import Field
        from ..simulation import Simulation
        from ..ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
except:
        try:
                #import classes from pyphasefield library
            from pyphasefield.field import Field
            from pyphasefield.simulation import Simulation
            from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE
        except:
            raise ImportError("Cannot import from pyphasefield library!")

def engine_CahnAllen(sim):
    dt = sim.dt #SIM是被传进来的对象，其实叫什么都行，dt是属性
    phi = sim.fields[0]
    M = sim.user_data["M"]
    W = sim.user_data["W"]

    delta_f = sim.user_data["delta_f"]
    epsilon = sim.user_data["epsilon"]
    deltaphi =(dt * M * (epsilon ** 2 * phi.laplacian() - 16. * W *
            (4. * phi.data ** 3 - 6. * phi.data ** 2 + 2. * phi.data))
            + delta_f * (30 * phi.data ** 2 - 60 * phi.data ** 3 + 30 * phi.data ** 4))
    sim.fields[0].data += deltaphi


def calc_phi(r, r0):
    return 0.5 * (1 - np.tanh((r - r0) / np.sqrt(2)))

class Homo_nucleation(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        if not ("M" in self.user_data):
            self.user_data["M"] = 1
        r0 = self.user_data["r0"] #user_data是这个类实例的属性之一。self关键字表示对当前对象实例的引用
        dim = self.dimensions
        phi = np.zeros(dim)
        x_center = dim[0] / 2
        y_center = dim[1] / 2
        Nx, Ny = dim
        for i in range(0, Nx):
            for j in range(0, Ny):
                r = np.sqrt(((i - x_center) ** 2 + (j - y_center) ** 2))
                phi[i][j] = calc_phi(r, r0)
        self.add_field(phi, "phi")

    def initialize_fields_and_imported_data(self):
        super().initialize_fields_and_imported_data()

    def just_before_simulating(self):
        super().just_before_simulating()
        # additional code to run just before beginning the simulation goes below
        # runs immediately before simulating, no manual changes permitted to changes implemented here

    def simulation_loop(self):
        engine_CahnAllen(self) #传递的是实例




        '''
        dt = self.dt
        phi = self.fields[0].data
        M = self.user_data["M"]
        W = self.user_data["W"]
        epsilon = self.user_data["epsilon"]


        delta_f = self.user_data["delta_f"]
        epsilon = self.user_data["epsilon"]
    
         laplacian_phi = laplace(phi) #

        deltaphi = dt * M * (epsilon ** 2 * laplacian_phi - 16. * W * \
                             (4. phi ** 3 - 6. * phi ** 2 + 2. * phi)) + \
                   delta_f * (30 * phi ** 2 - 60 * phi ** 3 - 30 * phi ** 4)

        phi += deltaphi
        '''



