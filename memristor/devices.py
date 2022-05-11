import yaml
import numpy as np
import pandas as pd
from pathlib import Path

path = Path(__file__).parent
with open(path / "params.yaml", 'r') as stream:
    CONFIG = yaml.safe_load(stream)
    PARAMS = CONFIG["StaticParameters"]

K_b = 1.38065e-23  # Boltzmann constant

HEADER = ["c0_set", "c1_set", "c2_set", "c3_set", "c4_set", "d0_set", "d1_set", "d2_set", "d3_set", "d4_set",
          "c0_reset", "c1_reset", "c2_reset", "c3_reset", "c4_reset", "d0_reset", "d1_reset", "d2_reset", "d3_reset",
          "d4_reset"]
DYNAMIC_PARAMS = pd.read_csv(path/'dynamic_params.txt', sep=" ", index_col=0, header=None)
DYNAMIC_PARAMS.columns = HEADER


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
        return np.random.normal(0, 4 * K_b * self.t * self.f * i_spacial / v, 1).item()

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
        i = i_spacial + self.temporal_variation(v, i_spacial)
        return i


class DynamicMemristor(StaticMemristor):
    def __init__(self, g_0):
        super().__init__(g_0)
        self.g_range = [-1,
                        -1]  # conduction range, [left_limit, right_limit], need this because parameters is calibrated
        # in terms of range. Set to dumb value first.
        self.params = None
        self.get_params()
        self.dynamic_d2d_var = np.random.normal(0, 1, 1).item()


    def get_params(self):
        assert 3.16 * 10e-6 <= self.g_0 <= 316 * 10e-6, "conductance out of range"
        if self.g_range[0] <= self.g_0 <= self.g_range[1]:
            return
        # 1. find the appropriate range
        # 2. get params based on range
        for index, row in DYNAMIC_PARAMS.iterrows():
            llimit, rlimit = index.split("-")
            llimit = float(llimit)*10e-6 # caste type and in us
            rlimit = float(rlimit)*10e-6 # caste type and in us
            if llimit <= self.g_0 <= rlimit:
                print('debug')
                self.g_range = [llimit, rlimit]
                self.params = row.to_dict()

    def set(self, V_p, t_p):
        self.get_params()
        logT = np.log(t_p)
        D_m = self.params["c0_set"] * (1-np.tanh(self.params["c1_set"]*(logT-self.params["c2_set"]))) \
             * (np.tanh(self.params["c3_set"]*V_p-self.params["c4_set"])+1)
        D_d2d = self.dynamic_d2d_var * D_m * (self.params["d0_set"] + self.params["d1_set"]*logT**2 +
                                              self.params["d2_set"]*V_p*logT + self.params["d3_set"]*(V_p**2)*logT +
                                              self.params["d4_set"]*V_p**3)
        self.g_0 += (D_m + D_d2d)
        if self.g_0 > 316 * 10e-6:
            self.g_0 = 316 * 10e-6

    def reset(self, V_p, t_p):
        self.get_params()
        logT = np.log(t_p)
        D_m = self.params["c0_reset"] * (-1 - np.tanh(self.params["c1_reset"] * (logT - self.params["c2_reset"]))) \
              * (np.tanh(self.params["c3_reset"] * V_p - self.params["c4_reset"]) - 1)
        D_d2d = self.dynamic_d2d_var * D_m * (self.params["d0_reset"] + self.params["d1_reset"] * logT ** 2 +
                                              self.params["d2_reset"] * V_p * logT + self.params["d3_reset"] * (
                                                          V_p ** 2) * logT +
                                              self.params["d4_reset"] * V_p ** 3)
        self.g_0 += (D_m + D_d2d)
        if 3.16 * 10e-6 > self.g_0:
            self.g_0 = 3.16 * 10e-6

# TODO: Qs for Amirali - T is Celcius or Kelvin? Role of frequency and appropriete value?
# TODO: Dynamic memristors
# TODO: Crossbar static
# TODO: Crossbar dynamic
