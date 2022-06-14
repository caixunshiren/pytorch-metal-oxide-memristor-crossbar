from ..devices import StaticMemristor, DynamicMemristor
import torch
import numpy as np


class LineResistanceCrossbar:
    """
    The crossbar model takes in voltage vector v and perform VMM with conductances W on the crossbar
        returns:  Wv
    This class does not:
        Normalize input. So all input should be normalized in the range for the memristor model
            e.g. for the static/dynamic memristor:
            3.16e-6 to 316e-6 for conductance
            -0.4 to 0.4 V for inference
        Normalize output.
    """
    def __init__(self, memristor_model, memristor_params, ideal_w):
        """
        :param memristor_model: memristor model class
        :param memristor_params: dictionary of the model param
        :param ideal_w: numpy/torch matrix of ideal conductances be programed
        """
        self.memristor_model = memristor_model
        self.memristor_params = memristor_params
        self.memristors = [[initialize_memristor(memristor_model, memristor_params, ideal_w[i, j])
                            for j in ideal_w.shape[1]] for i in ideal_w.shape[0]]
        self.ideal_w = ideal_w
        self.fitted_w = [[self.memristors[i][j].g_linfit for j in ideal_w.shape[1]] for i in ideal_w.shape[0]]
        self.cache = {}

    def ideal_vmm(self, v):
        """
        idealized vmm
        dims:
            v: bx1
            ideal_w: axb
        """
        return torch.matmul(self.ideal_w, v)

    def ideal_memristive_vmm(self, v):
        """
        vmm with non-ideal memristor inference and ideal crossbar
        dims:
            v: mx1
            crossbar: nxm
        """
        def mac_op(a1, a2):
            np.sum([a1[j].inference(a2[j]) for j in range(len(a1))])
        ret = torch.zeros([self.ideal_w.shape[0]])
        for i, row in enumerate(self.memristors):
            ret[i] = mac_op(row, v)
        return ret

    def lineres_memristive_vmm(self, v_dd):
        """
        vmm with non-ideal memristor inference and ideal crossbar
        dims:
            v_dd: mx1
            ideal_w: nxm
        """
        pass

    def solve_v(self, v_dd, W):
        """
        m word lines and n bit lines
        let M = [A, B; C, D]
        solve MV=E
        :param W: mxn matrix of conductances, type numpy array or nested list
        :return V: 2mn x 1 vector contains voltages of the word line and bit line
        """
        A = self.make_A(v_dd, W)
        B = self.make_B(v_dd, W)
        C = self.make_C(v_dd, W)
        D = self.make_D(v_dd, W)
        E = self.make_E(v_dd, W)
        M = torch.cat((torch.cat((A, B), 1), torch.cat((C, D), 1)), 0)
        M_inv = torch.inverse(M)
        self.cache["M_inv"] = M_inv
        return torch.matmul(M_inv, E)

    def make_E(self, W, v_dd):
        """
        :param W:
        :param v_dd:
        :return:
        """
        pass

    def make_A(self, W, v_dd):
        pass

    def make_B(self, W, v_dd):
        pass

    def make_C(self, W, v_dd):
        pass

    def make_D(self, W, v_dd):
        pass


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
        memristor
    else:
        raise Exception('Invalid memristor model')