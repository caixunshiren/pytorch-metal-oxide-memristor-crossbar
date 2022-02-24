import yaml
import numpy as np
from pathlib import Path

path = Path(__file__).parent
with open(path/"params.yaml", 'r') as stream:
    CONFIG = yaml.safe_load(stream)
    PARAMS = CONFIG["StaticParameters"]

K_b = 1.38065e-23  # Boltzmann constant


class StaticMemristor:
    def __init__(self, g_0):
        """
        :param g_0: device conductance in S, normally 1uS to 100 uS
        """
        self.g_0 = g_0
        self.t = None
        self.f = None
        self.u_A1 = None
        self.u_A3 = None
        self.sigma_A1 = None
        self.sigma_A3 = None
        self.d2d_var = np.random.normal(0, 1, 1).item()

    def noise_free_dc_iv_curve(self, v):
        return self.u_A1 * v + self.u_A3 * v ** 3

    def d2d_variation(self, v):
        return self.d2d_var * (self.sigma_A1 * v + self.sigma_A3 * v ** 3)

    def temporal_variation(self, v, i_spacial):
        return np.random.normal(0, 4*K_b*self.t*self.f*i_spacial/v, 1).item()

    def calibrate(self, t, f):
        self.t = t
        self.f = f
        self.u_A1 = PARAMS['A1']['a0'] + PARAMS['A1']['a1'] * self.g_0 + PARAMS['A1']['a2'] * self.t
        self.u_A3 = PARAMS['A3']['a0'] * self.g_0 + PARAMS['A3']['a1'] * self.g_0 ** 2 + \
                    PARAMS['A3']['a2'] * self.t ** (-1.33)
        self.sigma_A1 = PARAMS['A1']['p0'] + PARAMS['A1']['p1'] * self.g_0 + PARAMS['A1']['p2'] * self.t + \
                        PARAMS['A1']['p3'] * self.g_0 ** 2
        self.sigma_A3 = PARAMS['A3']['p0'] + PARAMS['A3']['p1'] * self.g_0 + PARAMS['A3']['p2'] * self.t + \
                        PARAMS['A3']['p3'] * self.g_0 ** 2 + PARAMS['A3']['p4'] * self.g_0 * self.t

    def inference(self, v):
        """
        :param v: applied voltage
        :param t: ambient temperature
        :param f: frequency
        :return: output current i
        """
        i_spacial = self.noise_free_dc_iv_curve(v) + self.d2d_variation(v)
        # print("debug 1:", i_spacial)
        i = i_spacial + self.temporal_variation(v, i_spacial)
        return i

# TODO: Qs for Amirali - T is Celcius or Kelvin? Role of frequency and appropriete value?
# TODO: Dynamic memristprs
# TODO: Crossbar static
# TODO: Crossbar dynamic
