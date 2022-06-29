from ..devices import StaticMemristor, DynamicMemristor, DynamicMemristorFreeRange, DynamicMemristorStuck
import torch
import numpy as np


class LineResistanceCrossbar:
    """
    The crossbar model takes in voltage vector v and perform VMM with conductances W on the crossbar
        returns:  Wv
    This class does not:
        Normalize input. So all input should be normalized in the range for the memristor model
            e.g. for the static/dynamic memristor:
            3.16e-6 to 316e-6 S for conductance
            -0.4 to 0.4 V for inference
        Normalize output.
    """
    def __init__(self, memristor_model, memristor_params, ideal_w, crossbar_params):
        """
        :param memristor_model: memristor model class
        :param memristor_params: dictionary of the model param
        :param ideal_w: nxm numpy/torch matrix of ideal conductances be programed
        """
        self.memristor_model = memristor_model
        self.memristor_params = memristor_params
        self.memristors = [[initialize_memristor(memristor_model, memristor_params, ideal_w[i, j])
                            for j in range(ideal_w.shape[1])] for i in range(ideal_w.shape[0])]
        self.ideal_w = ideal_w
        self.n, self.m = ideal_w.shape
        self.fitted_w = torch.tensor([[self.memristors[i][j].g_linfit for j in range(ideal_w.shape[1])]
                                      for i in range(ideal_w.shape[0])]).squeeze()
        self.cache = {}  # cache useful statistics to avoid redundant calculations

        self.V_SOURCE_MODE = crossbar_params["V_SOURCE_MODE"]

        # conductance of the word and bit lines.
        self.g_wl = torch.Tensor((1 / crossbar_params["r_wl"],))
        self.g_bl = torch.Tensor((1 / crossbar_params["r_bl"],))

        # WL & BL resistances and floating conductance
        self.r_in = crossbar_params["r_in"]
        self.r_out = crossbar_params["r_out"]
        self.g_floating = 1e-15

        if self.V_SOURCE_MODE == 'SINGLE_SIDE' or self.V_SOURCE_MODE == '|_':
            # line conductance of the sensor lines
            self.g_s_wl_in = torch.ones(self.m) / self.r_in
            self.g_s_wl_out = torch.ones(self.m) * self.g_floating  # floating
            self.g_s_bl_in = torch.ones(self.n) * self.g_floating  # floating
            self.g_s_bl_out = torch.ones(self.n) / self.r_out

        elif self.V_SOURCE_MODE == 'DOUBLE_SIDE' or self.V_SOURCE_MODE == '|=|':
            # line conductance of the sensor lines
            self.g_s_wl_in = torch.ones(self.m) / self.r_in
            self.g_s_wl_out = torch.ones(self.m) / self.r_in
            self.g_s_bl_in = torch.ones(self.n) / self.r_out
            self.g_s_bl_out = torch.ones(self.n) / self.r_out

        elif self.V_SOURCE_MODE == 'THREE_QUATER_SIDE' or self.V_SOURCE_MODE == '|_|':
            # line conductance of the sensor lines
            self.g_s_wl_in = torch.ones(self.m) / self.r_in
            self.g_s_wl_out = torch.ones(self.m) / self.r_in
            self.g_s_bl_in = torch.ones(self.n) * self.g_floating
            self.g_s_bl_out = torch.ones(self.n) / self.r_out

        else:
            raise ValueError("UNKOWN OPERATION MODE")

        # WL & BL voltages that are not the signal, assume bl_in, wl_out are tied low and bl_out is tied to 1 V.
        self.v_bl_in = torch.zeros(self.n)
        self.v_bl_out = torch.zeros(self.n)
        self.v_wl_out = torch.zeros(self.m)

    def recalibrate(self, i, j):
        """
        this function should be called every time the i,j th memristor get programmed
        :return:
        """
        self.memristors[i][j] = calibrate_memristor(self.memristor_model, self.memristors[i][j], self.memristor_params)
        self.ideal_w[i,j] = torch.tensor(self.memristors[i][j].g_0)
        self.fitted_w[i,j] = torch.tensor(self.memristors[i][j].g_linfit)

    def recalibrate_all(self):
        for i in range(self.n):
            for j in range(self.m):
                self.recalibrate(i,j)

    def ideal_vmm(self, v):
        """
        idealized vmm
        dims:
            v: mx1
            ideal_w: nxm
        """
        return torch.matmul(self.ideal_w, v)

    def naive_linear_memristive_vmm(self, v):
        """
        idealized vmm using fitted conductance of the memristors
        dims:
            v: mx1
            fitted_w: nxm
        """
        return torch.matmul(self.fitted_w, v)

    def naive_memristive_vmm(self, v):
        """
        vmm with non-ideal memristor inference and ideal crossbar
        dims:
            v: mx1
            crossbar: nxm
        """
        def mac_op(a1, a2):
            return torch.sum(torch.tensor([a1[j].inference(a2[j]) for j in range(len(a1))]))
        ret = torch.zeros([self.ideal_w.shape[0]])
        for i, row in enumerate(self.memristors):
            ret[i] = mac_op(row, v)
        return ret

    def lineres_memristive_vmm(self, v_wl_applied, v_bl_applied, iter=1, crossbar_cache=True, cap=True):
        """
        vmm with non-ideal memristor inference and ideal crossbar
        dims:
            v_dd: mx1
            ideal_w: nxm
        :param v_wl_applied: mx1 word line applied analog voltage
        :param v_bl_applied: nx1 bit line applied analog voltage
        :param iter: int. iter = 0 is constant conductance, iter = 1 is default first order g(v) approximation
                          iter = 2 is second order... and so on.
        :param crossbar_cache: whether cache useful statistics
        :param cap: if True, voltage will be capped at +-0.4 v for approximating conductance.
        :return: nx1 analog current vector
        """
        W = self.fitted_w
        V_crossbar = self.solve_v(W, v_wl_applied, v_bl_applied)
        for i in range(iter):
            V_crossbar = V_crossbar.view([-1, self.m, self.n])  # 2xmxn
            #print("debug", V_crossbar.shape, V_crossbar)
            V_wl, V_bl = torch.t(V_crossbar[0,:,:].squeeze()), torch.t(V_crossbar[1,:,:].squeeze())  # now nxm
            V_diff = V_wl - V_bl
            if cap:
                V_diff = torch.clamp(V_diff, min=-0.4, max=0.4)
            W = torch.tensor([[self.memristors[i][j].inference(V_diff[i,j]) for j in range(self.m)]
                              for i in range(self.n)])/V_diff
            V_crossbar = self.solve_v(W, v_wl_applied, v_bl_applied)
        V_crossbar = V_crossbar.view([-1, self.m, self.n])  # 2xmxn
        V_wl, V_bl = torch.t(V_crossbar[0, :, :].squeeze()), torch.t(V_crossbar[1, :, :].squeeze())  # now nxm
        V_diff = V_wl - V_bl
        if crossbar_cache:
            self.cache["V_wl"] = V_wl
            self.cache["V_bl"] = V_bl
        I = V_diff*W # nxm
        return torch.sum(I, dim=1)

    def solve_v(self, W, v_wl_applied, v_bl_applied):
        """
        m word lines and n bit lines
        let M = [A, B; C, D]
        solve MV=E
        :param W: nxm matrix of conductance, type torch tensor
        :param v_wl_applied: mx1 word line applied analog voltage
        :param v_bl_applied: nx1 bit line applied analog voltage
        :return V: 2mn x 1 vector contains voltages of the word line and bit line
        """
        A = self.make_A(W)
        B = self.make_B(W)
        C = self.make_C(W)
        D = self.make_D(W)
        E = self.make_E(v_wl_applied, v_bl_applied)
        M = torch.cat((torch.cat((A, B), 1), torch.cat((C, D), 1)), 0)
        M_inv = torch.inverse(M)
        self.cache["M_inv"] = M_inv
        return torch.matmul(M_inv, E)

    def make_E(self, v_wl_applied, v_bl_applied):
        m, n = self.m, self.n
        if self.V_SOURCE_MODE == 'SINGLE_SIDE' or self.V_SOURCE_MODE == '|_':
            E_B = torch.cat([torch.cat(((-self.v_bl_in[i] * self.g_s_bl_in[i]).view(1), torch.zeros(m-2), (-v_bl_applied[i] * self.g_s_bl_out[i]).view(1))).unsqueeze(1) for i in range(n)])
            E_W = torch.cat([torch.cat(((v_wl_applied[i] * self.g_s_wl_in[i]).view(1), torch.zeros(n-2), (self.v_wl_out[i] * self.g_s_wl_out[i]).view(1))) for i in range(m)]).unsqueeze(1)
            return torch.cat((E_W, E_B))
        elif self.V_SOURCE_MODE == 'DOUBLE_SIDE' or self.V_SOURCE_MODE == '|=|':
            E_B = torch.cat([torch.cat(((-v_bl_applied[i] * self.g_s_bl_in[i]).view(1), torch.zeros(m-2), (-v_bl_applied[i] * self.g_s_bl_out[i]).view(1))).unsqueeze(1) for i in range(n)])
            E_W = torch.cat([torch.cat(((v_wl_applied[i] * self.g_s_wl_in[i]).view(1), torch.zeros(n-2), (v_wl_applied[i] * self.g_s_wl_out[i]).view(1))) for i in range(m)]).unsqueeze(1)
            return torch.cat((E_W, E_B))
        elif self.V_SOURCE_MODE == 'THREE-QUATER_SIDE' or self.V_SOURCE_MODE == '|_|':
            E_B = torch.cat([torch.cat(((-self.v_bl_in[i] * self.g_s_bl_in[i]).view(1), torch.zeros(m-2), (-v_bl_applied[i] * self.g_s_bl_out[i]).view(1))).unsqueeze(1) for i in range(n)])
            E_W = torch.cat([torch.cat(((v_wl_applied[i] * self.g_s_wl_in[i]).view(1), torch.zeros(n-2), (v_wl_applied[i] * self.g_s_wl_out[i]).view(1))) for i in range(m)]).unsqueeze(1)
            return torch.cat((E_W, E_B))
        else:
            raise ValueError('UNKOWN OPERATION MODE')

    def make_A(self, W):
        W_t = torch.t(W)
        m, n = self.m, self.n

        def makea(i):
            return torch.diag(W_t[i, :]) \
                   + torch.diag(torch.cat((self.g_wl, self.g_wl * 2 * torch.ones(n - 2), self.g_wl))) \
                   + torch.diag(self.g_wl * -1 * torch.ones(n - 1), diagonal=1) \
                   + torch.diag(self.g_wl * -1 * torch.ones(n - 1), diagonal=-1) \
                   + torch.diag(torch.cat((self.g_s_wl_in[i].view(1), torch.zeros(n - 2), self.g_s_wl_out[i].view(1))))

        return torch.block_diag(*tuple(makea(i) for i in range(m)))

    def make_B(self, W):
        W_t = torch.t(W)
        m, n = self.m, self.n
        return torch.block_diag(*tuple(-torch.diag(W_t[i,:]) for i in range(m)))

    def make_C(self, W):
        W_t = torch.t(W)
        m, n = self.m, self.n

        def makec(j):
            return torch.zeros(m, m*n).index_put((torch.arange(m), torch.arange(m) * n + j), W_t[:, j])
            # c = torch.zeros(m, m*n)
            # for i in range(m):
            #     c[i, n*i+j] = W_t[i, j]
            # return c

        return torch.cat([makec(j) for j in range(n)],dim=0)

    def make_D(self, W):
        W_t = torch.t(W)
        m, n = self.m, self.n

        def maked(j):
            d = torch.zeros(m, m * n)

            i = 0
            d[i, j] = -self.g_s_bl_in[j] - self.g_bl - W_t[i, j]
            d[i, n * (i + 1) + j] = self.g_bl

            for i in range(1, m-1):
                d[i, n * (i - 1) + j] = self.g_bl
                d[i, n * i + j] = -self.g_bl - W_t[i, j] - self.g_bl
                d[i, n*(i+1)+j] = self.g_bl

            i = m - 1
            d[i, n * (i - 1) + j] = self.g_bl
            d[i, n * (i-0) + j] = -self.g_s_bl_out[j] - W_t[i, j] - self.g_bl

            return d

        return torch.cat([maked(j) for j in range(0,n)], dim=0)

    def lineres_memristive_programming(self, v_wl_applied, v_bl_applied, pulse_dur, iter=1, crossbar_cache=True, cap=True):
        """
        vmm with non-ideal memristor inference and ideal crossbar
        dims:
            v_dd: mx1
            ideal_w: nxm
        :param v_wl_applied: mx1 word line applied analog voltage
        :param v_bl_applied: nx1 bit line applied analog voltage
        :param pulse_dur: duration of the pulse in seconds
        :param iter: int. iter = 0 is constant conductance, iter = 1 is default first order g(v) approximation
                          iter = 2 is second order... and so on.
        :param crossbar_cache: whether cache useful statistics
        :param cap: if True, voltage will be capped at +-0.4 v for approximating conductance.
        :return: nx1 analog current vector
        """
        if self.memristor_model is not DynamicMemristor and self.memristor_model is not DynamicMemristorFreeRange:
            raise TypeError(self.memristor_model+' is not a programmable memristor type')
        W = self.fitted_w
        V_crossbar = self.solve_v(W, v_wl_applied, v_bl_applied)
        for i in range(iter):
            V_crossbar = V_crossbar.view([-1, self.m, self.n])  # 2xmxn
            # print("debug", V_crossbar.shape, V_crossbar)
            V_wl, V_bl = torch.t(V_crossbar[0, :, :].squeeze()), torch.t(V_crossbar[1, :, :].squeeze())  # now nxm
            V_diff = V_wl - V_bl
            if cap:
                V_diff = torch.clamp(V_diff, min=-0.4, max=0.4)
            W = torch.tensor([[self.memristors[i][j].inference(V_diff[i, j]) for j in range(self.m)]
                              for i in range(self.n)]) / V_diff
            V_crossbar = self.solve_v(W, v_wl_applied, v_bl_applied)
        V_crossbar = V_crossbar.view([-1, self.m, self.n])  # 2xmxn
        V_wl, V_bl = torch.t(V_crossbar[0, :, :].squeeze()), torch.t(V_crossbar[1, :, :].squeeze())  # now nxm
        V_diff = V_wl - V_bl
        for i in range(self.n):
            for j in range(self.m):
                if V_diff[i,j] > 0:
                    self.memristors[i][j].set(V_diff[i,j], pulse_dur)
                else:
                    self.memristors[i][j].reset(V_diff[i, j], pulse_dur)
                self.recalibrate(i,j)  # recalibrate the memristor at index i,j
        if crossbar_cache:
            self.cache["V_wl"] = V_wl
            self.cache["V_bl"] = V_bl


def initialize_memristor(memristor_model, memristor_params, g_0):
    """
    :param memristor_model: model to use
    :param memristor_params: parameter
    :param g_0: initial conductance
    :return: an unique calibrated memristor
    """
    if memristor_model == StaticMemristor:
        memristor = StaticMemristor(g_0)
        memristor.calibrate(memristor_params["temperature"], memristor_params["frequency"])
        return memristor
    elif memristor_model == DynamicMemristor:
        memristor = DynamicMemristor(g_0)
        memristor.calibrate(memristor_params["temperature"], memristor_params["frequency"])
        return memristor
    elif memristor_model == DynamicMemristorFreeRange:
        memristor = DynamicMemristorFreeRange(g_0)
        memristor.calibrate(memristor_params["temperature"], memristor_params["frequency"])
        return memristor
    elif memristor_model == DynamicMemristorStuck:
        memristor = DynamicMemristorStuck(g_0)
        memristor.calibrate(memristor_params["temperature"], memristor_params["frequency"])
        return memristor
    else:
        raise TypeError('Invalid memristor model')


def calibrate_memristor(memristor_model, memristor, memristor_params):
    """
    :param memristor_model: model to use
    :param memristor_params: parameter
    :param g_0: initial conductance
    :return: an unique calibrated memristor
    """
    if memristor_model == StaticMemristor:
        memristor.calibrate(memristor_params["temperature"], memristor_params["frequency"])
    elif memristor_model == DynamicMemristor:
        memristor.calibrate(memristor_params["temperature"], memristor_params["frequency"])
    elif memristor_model == DynamicMemristorFreeRange:
        memristor.calibrate(memristor_params["temperature"], memristor_params["frequency"])
    elif memristor_model == DynamicMemristorStuck:
        memristor.calibrate(memristor_params["temperature"], memristor_params["frequency"])
    else:
        raise TypeError('Invalid memristor model')
    return memristor