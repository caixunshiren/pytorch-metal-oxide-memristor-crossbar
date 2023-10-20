import matplotlib.pyplot as plt

from memristor.devices import StaticMemristor, DynamicMemristor, DynamicMemristorFreeRange, DynamicMemristorStuck
from memristor.crossbar.model import LineResistanceCrossbar
import torch
import numpy as np
from math import exp
from tqdm import tqdm


def graph_I_V(n, v_range, g_0, frequency, temperature):
    import matplotlib.pyplot as plt
    import numpy as np
    for j in range(n):
        memristor = StaticMemristor(g_0)
        memristor.calibrate(temperature, frequency)
        I = [memristor.inference(v) for v in np.linspace(v_range[0], v_range[1], 50)]
        I_linfit = [v * memristor.g_linfit for v in np.linspace(v_range[0], v_range[1], 50)]
        plt.plot(np.linspace(v_range[0], v_range[1], 50), I, label=f"simulation {j}")
        #plt.plot(np.linspace(v_range[0], v_range[1], 50), I_linfit, label=f"simulation {j} best fit")
    I_ohms_law = [v * g_0 for v in np.linspace(v_range[0], v_range[1], 50)]
    plt.plot(np.linspace(v_range[0], v_range[1], 50), I_ohms_law, label="Ohm's Law")
    plt.legend()
    plt.show()



def plot_conductance(iterations, g_0, t_p, v_p, temperature, frequency, OPERATION="SET"):
    import matplotlib.pyplot as plt
    memristor = DynamicMemristor(g_0)
    memristor.calibrate(temperature, frequency)
    conductances = [memristor.g_0]
    for i in range(iterations-1):
        if OPERATION == "SET":
            memristor.set(v_p, t_p)
        elif OPERATION == "RESET":
            memristor.reset(v_p, t_p)
        else:
            raise ValueError("UNKNOWN OPERATION")
        conductances.append(memristor.g_0)
    plt.plot(range(1, iterations+1), conductances)
    plt.show()


def plot_conductance_multiple(n, iterations, g_0, t_p, v_p, temperature, frequency, OPERATION="SET"):
    import matplotlib.pyplot as plt
    for j in range(n):
        memristor = DynamicMemristorFreeRange(g_0)
        memristor.calibrate(temperature, frequency)
        conductances = [memristor.g_0*1e6]
        for i in range(iterations-1):
            if OPERATION == "SET":
                memristor.set(v_p, t_p)
                plt.title("Conductance Change for SET")
            elif OPERATION == "RESET":
                memristor.reset(v_p, t_p)
                plt.title("Conductance Change for RESET")
            else:
                raise ValueError("UNKNOWN OPERATION")
            conductances.append(memristor.g_0*1e6)
        plt.plot(range(1, iterations+1), conductances)
    if OPERATION == "SET":
        plt.title("Conductance Change for SET")
        plt.text(0, 270, f'g_0: {g_0 * 1e6} uS\n t_p: {t_p * 1e3} ms\n v_p: {v_p} V', fontsize=11)
    elif OPERATION == "RESET":
        plt.title("Conductance Change for RESET")
        plt.text(0, 78, f'g_0: {g_0 * 1e6} uS\n t_p: {t_p * 1e3} ms\n v_p: {v_p} V', fontsize=11)
    plt.xlabel("pulse iteration")
    plt.ylabel("conductance (uS)")
    plt.show()


def plot_crossbar(crossbar, v_wl_applied, v_bl_applied):
    # plot output current
    out = torch.stack([crossbar.ideal_vmm(v_wl_applied), crossbar.naive_linear_memristive_vmm(v_wl_applied),
                        crossbar.naive_memristive_vmm(v_wl_applied),
                       crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied, iter=1)],
                      dim=1)
    out = torch.t(out)
    plt.matshow(out)
    plt.colorbar()
    plt.show()

    M = torch.t(crossbar.cache["V_wl"]-crossbar.cache["V_bl"])
    im = plt.imshow(M,
                    interpolation='none', aspect='equal')
    ax = plt.gca()
    # no need for axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Cell voltage for 32 x 64 passive TiOx crossbar")
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.colorbar()
    # plt.matshow(torch.t(crossbar.cache["V_wl"]-crossbar.cache["V_bl"]))
    # plt.grid(visible=True)
    plt.show()

    # word line voltage
    M = torch.t(crossbar.cache["V_wl"])
    im = plt.imshow(M,
                    interpolation='none', aspect='equal')
    ax = plt.gca()
    # no need for axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Word Line voltage for 32 x 64 passive TiOx crossbar")
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.colorbar()
    # plt.matshow(torch.t(crossbar.cache["V_wl"]-crossbar.cache["V_bl"]))
    # plt.grid(visible=True)
    plt.show()


    # bit line voltage
    M = torch.t(crossbar.cache["V_bl"])
    im = plt.imshow(M,
                    interpolation='none', aspect='equal')
    ax = plt.gca()
    # no need for axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Bit Line voltage for 32 x 64 passive TiOx crossbar")
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.colorbar()
    # plt.matshow(torch.t(crossbar.cache["V_wl"]-crossbar.cache["V_bl"]))
    # plt.grid(visible=True)
    plt.show()


    #crossbar conductance heatmap
    # idea conductance
    plt.matshow(torch.t(crossbar.ideal_w))
    plt.colorbar()
    plt.show()
    # best fit conductance
    plt.matshow(torch.t(crossbar.fitted_w))
    plt.colorbar()
    plt.show()

def plot_voltage_drop(crossbar, v_wl_applied, v_bl_applied):
    crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied, iter=1)
    M = torch.t(crossbar.cache["V_wl"]-crossbar.cache["V_bl"])
    im = plt.imshow(M,
                    interpolation='none', aspect='equal')
    ax = plt.gca()
    # no need for axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Cell voltage TiOx crossbar")
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.colorbar()
    # plt.matshow(torch.t(crossbar.cache["V_wl"]-crossbar.cache["V_bl"]))
    # plt.grid(visible=True)
    plt.show()

    # word line voltage
    M = torch.t(crossbar.cache["V_wl"])
    im = plt.imshow(M,
                    interpolation='none', aspect='equal')
    ax = plt.gca()
    # no need for axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Word Line voltage TiOx crossbar")
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.colorbar()
    # plt.matshow(torch.t(crossbar.cache["V_wl"]-crossbar.cache["V_bl"]))
    # plt.grid(visible=True)
    plt.show()


    # bit line voltage
    M = torch.t(crossbar.cache["V_bl"])
    im = plt.imshow(M,
                    interpolation='none', aspect='equal')
    ax = plt.gca()
    # no need for axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Bit Line voltage TiOx crossbar")
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.colorbar()
    # plt.matshow(torch.t(crossbar.cache["V_wl"]-crossbar.cache["V_bl"]))
    # plt.grid(visible=True)
    plt.show()


def plot_program_crossbar(crossbar, v_wl_applied, v_bl_applied, t_p, iterations):
    #crossbar conductance heatmap
    # idea conductance
    plt.matshow(torch.t(crossbar.ideal_w))
    plt.title("ideal conductance before programming")
    plt.colorbar()
    plt.show()
    # best fit conductance
    plt.matshow(torch.t(crossbar.fitted_w))
    plt.title("fitted conductance before programming")
    plt.colorbar()
    plt.show()

    ideal_w_hist=[torch.clone(crossbar.ideal_w).view(-1)]
    fitted_w_hist=[torch.clone(crossbar.fitted_w).view(-1)]
    for i in range(iterations):
        #  program the crossbar
        crossbar.lineres_memristive_programming(v_wl_applied, v_bl_applied, t_p)
        ideal_w_hist.append(torch.clone(crossbar.ideal_w).view(-1))
        fitted_w_hist.append(torch.clone(crossbar.fitted_w).view(-1))

    #crossbar conductance heatmap
    # idea conductance
    plt.matshow(torch.t(crossbar.ideal_w))
    plt.title("ideal conductance after programming")
    plt.colorbar()
    plt.show()
    # best fit conductance
    plt.matshow(torch.t(crossbar.fitted_w))
    plt.title("fitted conductance after programming")
    plt.colorbar()
    plt.show()

    # conductance change history
    #print(ideal_w_hist)
    ideal_w_hist = torch.stack(ideal_w_hist, dim=1)
    fitted_w_hist = torch.stack(fitted_w_hist, dim=1)
    #print(ideal_w_hist)
    for i in range(ideal_w_hist.shape[0]):
        plt.plot(range(iterations + 1), ideal_w_hist[i, :])
    plt.title("ideal conductance change")
    plt.show()

    for i in range(fitted_w_hist.shape[0]):
        plt.plot(range(iterations + 1), fitted_w_hist[i, :])
    plt.title("fitted conductance change")
    plt.show()

def fig1():
    frequency = 1e8  # hz
    temperature = 60+273  # Kelvin
    g_0 = 50e-6  # S
    v = 0.3  # V
    memristor = StaticMemristor(g_0)
    memristor.calibrate(temperature, frequency)
    for i in range(10):
        print(memristor.inference(v))
    ideal_i = v * g_0
    print("ideal naive linear estimate:", ideal_i)
    print("ideal naive non-linear estimate:", memristor.noise_free_dc_iv_curve(v))

    graph_I_V(10, [-0.4, 0.4], g_0, frequency, temperature)


def fig2():
    v_p = 1.0 # range [−0.8 V to −1.5 V]/[0.8 V to 1.15 V]
    t_p = 0.5e-3 # programming pulse duration
    g_0 = 60e-6
    frequency = 1e8  # hz
    temperature = 273 + 60  # Kelvin
    plot_conductance_multiple(20, 100, g_0, t_p, v_p, temperature, frequency, OPERATION="SET")


def fig3():
    torch.set_default_dtype(torch.float64)

    crossbar_params = {'r_wl': 20, 'r_bl': 20, 'r_in':10, 'r_out':10, 'V_SOURCE_MODE':'|_|'}
    memristor_model = StaticMemristor
    memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
    #ideal_w = torch.tensor([[50, 100],[75, 220],[30, 80]], dtype=torch.float64)*1e-6
    ideal_w = torch.FloatTensor(48, 16).uniform_(10, 300).double()*1e-6

    crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)
    #v_applied = torch.tensor([-0.2, 0.3], dtype=torch.float64)
    v_wl_applied = torch.concat([1.8*torch.ones(4,), torch.linspace(1.8, 1.4,12)], dim=0)#torch.FloatTensor(32,).uniform_(-0.4, 0.4).double()
    v_bl_applied = 0*torch.concat([torch.linspace(1.6, 2.7,16), 2.7*torch.ones(16,),torch.linspace(2.7, 1.6,16)], dim=0) #1.7*torch.ones(48,)#torch.zeros(32, )

    #print("ideal vmm:", crossbar.ideal_vmm(v_applied))
    #print("naive linear memristive vmm:", crossbar.naive_linear_memristive_vmm(v_applied))
    #print("naive memristive vmm:", crossbar.naive_memristive_vmm(v_applied))
    #print("line resistance memristive vmm:", crossbar.lineres_memristive_vmm(v_applied, order=1))
    plot_voltage_drop(crossbar, v_wl_applied, v_bl_applied)


def fig4():
    torch.set_default_dtype(torch.float64)

    crossbar_params = {'r_wl': 10, 'r_bl': 10, 'r_in':10, 'r_out':10, 'V_SOURCE_MODE':'|=|'}
    memristor_model = DynamicMemristorFreeRange
    memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
    #ideal_w = torch.tensor([[50, 100],[75, 220],[30, 80]], dtype=torch.float64)*1e-6
    #ideal_w = torch.FloatTensor(48, 16).uniform_(10, 300).double()*1e-6
    ideal_w = 200*torch.ones(48, 16)*1e-6

    crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)
    #v_applied = torch.tensor([-0.2, 0.3], dtype=torch.float64)
    v_wl_applied = 0*torch.ones(16,)#torch.FloatTensor(32,).uniform_(-0.4, 0.4).double()
    #v_bl_applied = 0.4*torch.ones(48,)#torch.zeros(32, )
    v_bl_applied = torch.concat([torch.linspace(1.5, 2.5,16), 2.5*torch.ones(16,),torch.linspace(2.5, 1.5,16)], dim=0)
    t_p = 0.5e-3 # programming pulse duration
    iter = 20

    plot_program_crossbar(crossbar, v_wl_applied, v_bl_applied, t_p, iter)


class Component:
    def apply(self, *args, **kwargs):
        return


class CurrentDecoder(Component):
    """
    binary decoder based on a threshold vector t
    """
    def calibrate_t(self, crossbar: LineResistanceCrossbar, itr: int = 50) -> torch.Tensor:
        m, n = crossbar.m, crossbar.n
        X = 0.3*(torch.randint(low=0, high=2, size=[m, itr]))+0.1
        t = torch.zeros([n,])
        for i in tqdm(range(itr)):
            t = t + crossbar.lineres_memristive_vmm(X[:, i], torch.zeros([n]))
        return t/itr

    def apply(self, x, t=None) -> torch.Tensor:
        if t is None:
            t = torch.mean(x) * torch.ones(x.shape)
        return torch.ge(x, t).type(torch.float64)

    def calibrate_max_current(self, crossbar: LineResistanceCrossbar, n_reset: int = 4000, itr: int = 1) -> torch.Tensor:
        """
        Find the maximum possible current by the crossbar (minimum would be 0)
        Used for ADC decoder calibration
        Must re-program the crossbar after calibration
        """
        m, n = crossbar.m, crossbar.n
        t_p_reset = 100e-3

        # Set all weights to LRS
        # V_diff is positive to SET the memristor
        # so a negative v_p_bl will SET the memristor (with 0 v_p_wl)
        # We use 1.5V to SET the memristor
        v_p_bl = -1.8 * torch.ones(n, )
        for j in tqdm(range(n_reset)):
            crossbar.lineres_memristive_programming(torch.zeros(m, ), v_p_bl, t_p_reset, order=2, cap=True, log_power=True)

        # Set all inputs to 1 for maximum current
        # assuming that inference uses 0.4V
        X = torch.ones(size=[m, itr]) * 0.4
        t_max = torch.zeros([n, ])
        for i in tqdm(range(itr)):
            out = crossbar.lineres_memristive_vmm(X[:, i], torch.zeros([n]))
            t_max = t_max + out
            print("calibration output", out)

        return t_max / itr
    
    def calibrate_min_current(self, crossbar: LineResistanceCrossbar, n_reset: int = 4000, itr: int = 1) -> torch.Tensor:
        """
        Find the maximum possible current by the crossbar (minimum would be 0)
        Used for ADC decoder calibration
        Must re-program the crossbar after calibration
        """
        m, n = crossbar.m, crossbar.n
        t_p_reset = 100e-3

        # Set all weights to LRS
        # V_diff is positive to SET the memristor
        # so a negative v_p_bl will SET the memristor (with 0 v_p_wl)
        # We use 1.5V to SET the memristor

        v_p_bl = 1.8 * torch.ones(n, )
        for j in tqdm(range(n_reset)):
            crossbar.lineres_memristive_programming(torch.zeros(m, ), v_p_bl, t_p_reset, order=2, cap=True, log_power=True)

        # Set all inputs to 1 for maximum current
        # assuming that inference uses 0.4V
        X = torch.ones(size=[m, itr]) * 0.4
        t_min = torch.zeros([n, ])
        for i in tqdm(range(itr)):
            out = crossbar.lineres_memristive_vmm(X[:, i], torch.zeros([n]))
            t_min = t_min + out
            print("calibration output", out)

        return t_min / itr

    def apply_2_bits(self, x, min_current, max_current) -> torch.Tensor:
        thresholds = torch.linspace(0, 1, 5)[1:-1].unsqueeze(-1)
        min_current = min_current.reshape(1, -1)
        max_current = max_current.reshape(1, -1)
        thresholds = thresholds * (max_current - min_current) + min_current
        output = torch.zeros_like(x)
        print("raw output", x)
        for threshold in thresholds:
            output += torch.ge(x, threshold).type(torch.float64)
        print("output", output)
        return output

    def calibrate_binary_crossbar_output_current_thresholds(self,
                                                            crossbar: LineResistanceCrossbar,
                                                            binary_crossbar: torch.Tensor,
                                                            itr: int = 10):
        """
        Create all possible input combinations, manually calculate the output value for each input
        Then we map the output current to the binary crossbar output
        If an output cannot be reached, we don't map it.
        we repeat the process itr times and take the average
        This is so the decoder can directly map the output current to the binary crossbar output, and ignore outputs
        that doesn't exist

        """
        torch.set_default_dtype(torch.float64)
        m, n = crossbar.m, crossbar.n
        # crossbar_bit_lines: {bit_line_index: {possible_value: [sum_current_over_iterations, number_of_occurrence]}}
        crossbar_bit_lines = {i: {} for i in range(n)}
        # Create all possible input combinations
        X = torch.zeros(size=[m, ])
        for i in range(itr):
            output = torch.matmul(X, binary_crossbar).squeeze() # [n, ]
            output_current = crossbar.lineres_memristive_vmm(0.4 * X, torch.zeros([n]))
            for j in range(n):
                # if the output current is nan, treat it as 0
                if torch.isnan(output_current[j]):
                    continue
                if output[j].item() not in crossbar_bit_lines[j]:
                    crossbar_bit_lines[j][output[j].item()] = [output_current[j].item(), 1]
                else:
                    crossbar_bit_lines[j][output[j].item()][0] += output_current[j].item()
                    crossbar_bit_lines[j][output[j].item()][1] += 1
            while X.sum() < m:
                index = m - 1
                while X[index] == 1:
                    X[index] = 0
                    index -= 1
                X[index] = 1
                output = torch.matmul(X, binary_crossbar).squeeze() # [n, ]
                output_current = crossbar.lineres_memristive_vmm(0.4 * X, torch.zeros([n]))
                for j in range(n):
                    # if the output current is nan, treat it as 0
                    if torch.isnan(output_current[j]):
                        continue
                    if output[j].item() not in crossbar_bit_lines[j]:
                        crossbar_bit_lines[j][output[j].item()] = [output_current[j].item(), 1]
                    else:
                        crossbar_bit_lines[j][output[j].item()][0] += output_current[j].item()
                        crossbar_bit_lines[j][output[j].item()][1] += 1
        # Calculate the average output current for each output value
        for j in range(n):
            for output in crossbar_bit_lines[j]:
                crossbar_bit_lines[j][output] = crossbar_bit_lines[j][output][0] / crossbar_bit_lines[j][output][1]
        return crossbar_bit_lines


    def decode_binary_crossbar_output(self, crossbar_bit_lines: dict, output_current: torch.Tensor):
        """
        Given the output current, map it to the binary crossbar output that is closest to the output current
        """
        output = torch.zeros_like(output_current)
        for j in range(output_current.shape[0]):
            min_diff = float('inf')
            # if output is nan, treat it as 0
            if torch.isnan(output_current[j]):
                output[j] = 0
                continue
            for possible_output in crossbar_bit_lines[j]:
                diff = abs(crossbar_bit_lines[j][possible_output] - output_current[j])
                if diff < min_diff:
                    min_diff = diff
                    output[j] = possible_output
        # print("\tdecoded output", output)
        return output

def test_inference():
    torch.set_default_dtype(torch.float64)

    crossbar_params = {'r_wl': 20, 'r_bl': 20, 'r_in': 10, 'r_out': 10, 'V_SOURCE_MODE': '|_|'}
    memristor_model = DynamicMemristorStuck
    memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
    # ideal_w = torch.tensor([[50, 100],[75, 220],[30, 80]], dtype=torch.float64)*1e-6
    ideal_w = torch.ones([48, 16])*65e-6

    crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)
    # randomize weights
    n_reset = 40
    t_p_reset = 0.5e-3
    v_p_bl = 1.5 * torch.cat([torch.linspace(1, 1.2, 16), 1.2 * torch.ones(16, ),
                              torch.linspace(1.2, 1, 16)], dim=0)
    print("V_p_bl", v_p_bl)
    for j in tqdm(range(n_reset)):
        crossbar.lineres_memristive_programming(torch.zeros(16, ), v_p_bl, t_p_reset)

    decoder = CurrentDecoder()
    v_wl_applied = 0.3*(torch.randint(low=0, high=2, size=[16,]))+0.1
    print("input", v_wl_applied)
    v_bl_applied = torch.zeros(48)
    t = decoder.calibrate_t(crossbar, 100)
    print("threshold", t)
    x = crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied)
    print("raw output", x)
    print("naive binarization", decoder.apply(x))
    print("fitted binarization", decoder.apply(x, t))

def test_power():
    torch.set_default_dtype(torch.float64)
    crossbar_params = {'r_wl': 20, 'r_bl': 20, 'r_in': 10, 'r_out': 10, 'V_SOURCE_MODE': '|_'}
    memristor_model = DynamicMemristorStuck
    memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
    ideal_w = torch.ones([48, 16])*65e-6
    crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)
    # randomize weights
    n_reset = 40
    t_p_reset = 0.5e-3
    v_p_bl = 1.5 * torch.cat([torch.linspace(1, 1.2, 16), 1.2 * torch.ones(16, ),
                              torch.linspace(1.2, 1, 16)], dim=0)
    for j in tqdm(range(n_reset)):
        crossbar.lineres_memristive_programming(torch.zeros(16, ), v_p_bl, t_p_reset, log_power=True)

    decoder = CurrentDecoder()
    v_wl_applied = 0.3*(torch.randint(low=0, high=2, size=[16,]))+0.1
    print("input", v_wl_applied)
    v_bl_applied = torch.zeros(48)
    x = crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied, log_power=True)
    print("power:")
    for ticket in crossbar.power_log:
        print(f"{ticket.name} - {ticket.op_type}")
        print("Total Power:", ticket.power_total)
        print("Memristor Power:", ticket.power_memristor)
        print("Word Line Power:", ticket.power_wordline)
        print("Bit Line Power:", ticket.power_bitline)


def build_binary_matrix_crossbar(binary_weights: torch.Tensor, n_reset: int = 128, t_p_reset = 100e-3, set_voltage_difference = 1.8) -> LineResistanceCrossbar:
    """
    Given the input of a binary matrix, build a crossbar with the same shape and conductance
    :param binary_weights: binary matrix (m, n) where m is the number of rows and n is the number of columns. rows are wordlines (input) and columns are bitlines (output)
    :param n_reset: number of reset pulses
    :param t_p_reset: reset pulse duration
    :return: crossbar with the same shape and conductance
    """
    torch.set_default_dtype(torch.float64)
    crossbar_params = {'r_wl': 20, 'r_bl': 20, 'r_in': 10, 'r_out': 10, 'V_SOURCE_MODE': '|_|'}
    memristor_model = DynamicMemristorStuck
    memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
    m, n = binary_weights.shape
    ideal_w = torch.ones([n, m])*65e-6  # shape is [number_of_cols, number_of_rows]
    crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)

    for _ in tqdm(range(n_reset)):
        for i in range(m):
            for j in range(n):
                bit = binary_weights[i, j]
                v_p_wl = torch.zeros(m)
                v_p_bl = torch.zeros(n)
                if bit == 1:
                    v_p_wl[i] = set_voltage_difference/2
                    v_p_bl[j] = -set_voltage_difference/2
                else:
                    v_p_wl[i] = -set_voltage_difference/2
                    v_p_bl[j] = set_voltage_difference/2
                crossbar.lineres_memristive_programming(v_p_wl, v_p_bl, t_p_reset, order=2, log_power=True)
    return crossbar


def build_binary_matrix_crossbar_split_into_subsections(
        binary_weights: torch.Tensor,
        num_row_splits: int = 1, num_col_splits: int = 1,
        n_reset: int = 128, t_p_reset=100e-3, set_voltage_difference=1.8):
    """
    Given the input of a binary matrix, build a crossbar with the same shape and conductance
    This specific function also splits the crossbar into subsections - for easy programming
    :param binary_weights: binary matrix (m, n) where m is the number of rows and n is the number of columns. rows are wordlines (input) and columns are bitlines (output)
    :param num_row_splits: number of row splits
    :param num_col_splits: number of column splits
    :param n_reset: number of reset pulses
    :param t_p_reset: reset pulse duration
    :param set_voltage_difference: voltage difference between the two states
    :return: a 2-D list of crossbars with the same shape and conductance when combined, and a 2-D list of decoders
    """
    torch.set_default_dtype(torch.float64)
    crossbar_params = {'r_wl': 20, 'r_bl': 20, 'r_in': 20, 'r_out': 20, 'V_SOURCE_MODE': '|_|'}
    memristor_model = DynamicMemristorStuck
    memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
    m, n = binary_weights.shape

    # full_ideal_w = torch.ones([n, m]) * 65e-6  # shape is [number_of_cols, number_of_rows]
    # This time we initialize ideal_w to be what we want
    # full_ideal_w = binary_weights transpose, from 3.16e-6 <= self.g_0 <= 316e-6
    full_ideal_w = binary_weights.transpose(0, 1).clone() * (316e-6 - 3.16e-6) + 3.16e-6
    import math
    num_rows_per_split = math.ceil(m / num_row_splits)
    num_cols_per_split = math.ceil(n / num_col_splits)
    crossbars_2d_list = []

    decoder_dicts_2d_list = []

    decoder = CurrentDecoder()

    for row_split in range(num_row_splits):
        crossbars_1d_list = []
        decoder_dicts_1d_list = []
        for col_split in range(num_col_splits):
            ideal_w = full_ideal_w[col_split*num_cols_per_split:(col_split+1)*num_cols_per_split, row_split*num_rows_per_split:(row_split+1)*num_rows_per_split]
            crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)

            for _ in tqdm(range(n_reset)):
                for i in range(min(num_rows_per_split, m-row_split*num_rows_per_split)):
                    for j in range(min(num_cols_per_split, n-col_split*num_cols_per_split)):
                        bit = binary_weights[row_split*num_rows_per_split+i, col_split*num_cols_per_split+j]
                        v_p_wl = torch.zeros(min(num_rows_per_split, m-row_split*num_rows_per_split))
                        v_p_bl = torch.zeros(min(num_cols_per_split, n-col_split*num_cols_per_split))
                        if bit == 1:
                            v_p_wl[i] = set_voltage_difference/2
                            v_p_bl[j] = -set_voltage_difference/2
                        else:
                            v_p_wl[i] = -set_voltage_difference/2
                            v_p_bl[j] = set_voltage_difference/2
                        crossbar.lineres_memristive_programming(v_p_wl, v_p_bl, t_p_reset, order=10, log_power=True)
            decoder_dict = decoder.calibrate_binary_crossbar_output_current_thresholds(crossbar, binary_weights[row_split*num_rows_per_split:(row_split+1)*num_rows_per_split, col_split*num_cols_per_split:(col_split+1)*num_cols_per_split])
            crossbars_1d_list.append(crossbar)
            decoder_dicts_1d_list.append(decoder_dict)
        crossbars_2d_list.append(crossbars_1d_list)
        decoder_dicts_2d_list.append(decoder_dicts_1d_list)
    return crossbars_2d_list, decoder_dicts_2d_list


def calculate_vmm_result_split_into_subsection(
        crossbars_2d_list: list,
        bit_line_possible_outputs: list,
        decoder: CurrentDecoder,
        input_vector: torch.Tensor, twos_complement: bool = True,
        debug_weight_matrix: torch.Tensor = None):
    """
    Given a 2-D list of crossbars, calculate the VMM result
    :param crossbars_2d_list: a 2-D list of crossbars
    :param bit_line_possible_outputs: a 2-D list of possible outputs for each bitline - corresponds to the crossbars_2d_list
    :param decoder: a decoder object
    :param input_vector: input vector
    :param twos_complement: whether the input vector is in two's complement
    :return: the VMM result - an integer
    """
    length_of_input_vector = input_vector.shape[1]

    final_result = 0
    # calculate the end result of each bit of input (in sequence, with the first bit being the MSB)
    for bit in range(length_of_input_vector):
        # Keep track the last row we processed, so we input the correct input vector corresponding to the row we want
        last_row = 0
        # Bit result is 0 by default
        bit_result = 0
        # Iterate through each row split
        for row_split in range(len(crossbars_2d_list)):
            # The number of rows in the crossbar of this row split. This is just default value, it will be updated later
            m = crossbars_2d_list[row_split][0].m
            # col_count is for shift and add. For each row split, we start from the rightmost column
            col_count = 0
            # Get the input vector corresponding to the crossbar, make sure it's for the correct row
            # Range of value is from last_row to last_row + m
            v_wl_applied = input_vector[last_row:last_row + m, bit] * 0.4
            # Iterate through each column split (col_split will be used as index of crossbar, but from right to left)
            for col_split in range(len(crossbars_2d_list[row_split])):
                # we reverse col_split so that we can iterate from right to left
                col_split_index = len(crossbars_2d_list[row_split]) - col_split - 1
                # Get the crossbar
                crossbar = crossbars_2d_list[row_split][col_split_index]
                # Get the shape of the crossbar
                m, n = crossbar.m, crossbar.n
                # input vector to bit lines are always 0, but has shape corresponding to each crossbar
                v_bl_applied = torch.zeros(n)
                # Calculate the output of the crossbar
                x = crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied, log_power=True)
                # Decode the output of the crossbar
                decoded = decoder.decode_binary_crossbar_output(bit_line_possible_outputs[row_split][col_split_index], x)
                if debug_weight_matrix is not None:
                    current_weight = debug_weight_matrix[last_row:last_row+m, len(debug_weight_matrix[last_row])-(col_count + crossbar.n):len(debug_weight_matrix[last_row])-(col_count + crossbar.n)+crossbar.n]
                    # print("weight matrix:\n", current_weight)
                    # print("input vector:", v_wl_applied)
                    # print("decoded output", decoded)
                    for decoded_bit in range(len(decoded)):
                        if decoded[decoded_bit].item() != sum((v_wl_applied/0.4) * current_weight[:, decoded_bit].reshape(-1)).item():
                            print("=========== BIT IS WRONG ===========")
                            print("\tExpected:", sum((v_wl_applied/0.4) * current_weight[:, decoded_bit].reshape(-1)).item())
                            print("\tActual:", decoded[decoded_bit].item())
                            print("\tInput vector:", v_wl_applied)
                            print("\tWeight matrix:\n", current_weight)
                            print("\tDecoded output:", decoded)
                # Shift and add the decoded output to the bit result
                # Iterate through each decoded bit (this is from left to right for each crossbar)
                for i, decoded_bit in enumerate(decoded):
                    # If left most crossbar, and is first bit, and we use 2's complement, then we need to subtract
                    if col_split_index == 0 and i == 0 and twos_complement:
                        # decoded bit is shifted by multiplying 2 to the power of (col_count + crossbar.n - i)
                        # where col_count is the total number of bits we have processed before this crossbar
                        # col_count + crossbar.n is the total number of bits we will have processed after this crossbar
                        # col_count + crossbar.n - i is the current bit's position
                        # subtract 1 to make sure the rightmost bit is 2^0
                        bit_result -= decoded_bit * 2 ** (col_count + crossbar.n - i - 1)
                    else:
                        bit_result += decoded_bit * 2 ** (col_count + crossbar.n - i - 1)
                # Update the col_count, so when we process one crossbar to the left we record num bits already processed
                col_count += crossbar.n
            # Update the last row we processed, so we get correct input vector for the next row split
            last_row += m
        # Add the bit result to the final result
        if bit == 0 and twos_complement:
            final_result -= bit_result * 2 ** (length_of_input_vector - bit - 1)
        else:
            final_result += bit_result * 2 ** (length_of_input_vector - bit - 1)
    return final_result.item()

def test_sequential_bit_input_inference_and_power():
    torch.set_default_dtype(torch.float64)
    weights = torch.tensor([
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1]
    ], dtype=torch.float64)

    # IMPORTANT: 1024 reset pulses are needed to ensure that the crossbar is fully reset
    # This has been tested, where smaller number of reset pulses result in incorrect inference
    crossbar = build_binary_matrix_crossbar(weights,n_reset = 5, t_p_reset=20)
    decoder = CurrentDecoder()
    bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(crossbar, weights)

    for number_1 in range(1, 11):
        for number_2 in range(1, 11):
            for number_3 in range(1, 11):
                binary_1 = "0" * (8 - len(bin(number_1)[2:])) + bin(number_1)[2:]
                binary_2 = "0" * (8 - len(bin(number_2)[2:])) + bin(number_2)[2:]
                binary_3 = "0" * (8 - len(bin(number_3)[2:])) + bin(number_3)[2:]
                binary_array_1 = [int(bit) for bit in binary_1]
                binary_array_2 = [int(bit) for bit in binary_2]
                binary_array_3 = [int(bit) for bit in binary_3]
                input_vector = torch.tensor([
                    binary_array_1,
                    binary_array_2,
                    binary_array_3
                ], dtype=torch.float64)
                print("input numbers", number_1, number_2, number_3)
                print("expected output", number_1*5 + number_2*3 + number_3*3)
                final_result = 0
                for bit in range(8):
                    v_wl_applied = input_vector[:, bit] * 0.4
                    v_bl_applied = torch.zeros(8)
                    # print("\tinput", v_wl_applied)
                    x = crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied, log_power=True)
                    # shift and add
                    decoded = decoder.decode_binary_crossbar_output(bit_line_possible_outputs, x)
                    bit_result = 0
                    for i, decoded_bit in enumerate(decoded):
                        bit_result += decoded_bit * 2 ** (8 - i - 1)
                    # print("\tbit result", bit_result)
                    final_result += bit_result * 2 ** (8 - bit - 1)
                print("final result", final_result)


def calculate_vmm_result(crossbar: LineResistanceCrossbar, bit_line_possible_outputs: dict, decoder: CurrentDecoder, input_vector: torch.Tensor, twos_complement: bool = True):
    """
    Calculate the result of a vector matrix multiplication
    :param crossbar: crossbar that is pre-programmed with the weights
    :param bit_line_possible_outputs: dictionary of possible outputs for each bitline for decoding purpose
    :param decoder: decoder used to process current into digital output
    :param input_vector: input vector (binary numbers, 2d tensor) to be multiplied with the weights
    :return: the result of the vector matrix multiplication as a decimal number
    """
    length_of_input_vector = input_vector.shape[1]
    m, n = crossbar.m, crossbar.n
    final_result = 0
    for bit in range(length_of_input_vector):
        v_wl_applied = input_vector[:, bit] * 0.4
        v_bl_applied = torch.zeros(n)
        x = crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied, log_power=True)
        # shift and add
        decoded = decoder.decode_binary_crossbar_output(bit_line_possible_outputs, x)
        bit_result = 0
        for i, decoded_bit in enumerate(decoded):
            if i == 0 and twos_complement:
                bit_result -= decoded_bit * 2 ** (n - i - 1)
            else:
                bit_result += decoded_bit * 2 ** (n - i - 1)
        if twos_complement and bit == 0:
            final_result -= bit_result * 2 ** (length_of_input_vector - bit - 1)
        else:
            final_result += bit_result * 2 ** (length_of_input_vector - bit - 1)
    return final_result.item()


def calculate_vmm_result_left_and_right(
        left_crossbar: LineResistanceCrossbar, right_crossbar: LineResistanceCrossbar,
        left_bit_line_possible_outputs: dict, right_bit_line_possible_outputs: dict,
        decoder: CurrentDecoder,
        input_vector: torch.Tensor, twos_complement: bool = True):
    length_of_input_vector = input_vector.shape[1]
    m, n_left, n_right = input_vector.shape[0], left_crossbar.n, right_crossbar.n

    # assume left crossbar contains MSB

    final_result = 0
    for bit in range(length_of_input_vector):
        v_wl_applied = input_vector[:, bit] * 0.4
        v_bl_applied_left = torch.zeros(n_left)
        v_bl_applied_right = torch.zeros(n_right)
        left_x = left_crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied_left, log_power=True)
        right_x = right_crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied_right, log_power=True)
        # shift and add
        left_decoded = decoder.decode_binary_crossbar_output(left_bit_line_possible_outputs, left_x)
        right_decoded = decoder.decode_binary_crossbar_output(right_bit_line_possible_outputs, right_x)
        bit_result = 0

        for i, decoded_bit in enumerate(left_decoded):
            if i == 0 and twos_complement:
                bit_result -= decoded_bit * 2 ** (n_left+n_right - i - 1)
            else:
                bit_result += decoded_bit * 2 ** (n_left+n_right - i - 1)

        for i, decoded_bit in enumerate(right_decoded):
            bit_result += decoded_bit * 2 ** (n_right - i - 1)

        if twos_complement and bit == 0:
            final_result -= bit_result * 2 ** (length_of_input_vector - bit - 1)
        else:
            final_result += bit_result * 2 ** (length_of_input_vector - bit - 1)
    return final_result.item()


def convert_to_binary_array(number, int_bits, fraction_bits, adjust_for_multiplication=True):
    """
    Convert a decimal number to a binary array
    Consider 2's complement for negative numbers
    :param number: decimal number
    :param int_bits: number of integer bits
    :param fraction_bits: number of fraction bits
    :param adjust_for_multiplication: whether to adjust the number for multiplication (add leading 1s or 0s)
    :return: binary array
    """
    number = round((number * 2 ** fraction_bits).item())
    if number >= 0:
        binary_string = bin(number)[2:].zfill(int_bits + fraction_bits)  # Convert integer to binary and fill leading zeros
        binary_array = [int(bit) for bit in binary_string]
        # Add same number of bits of zeros to the beginning of the array
        if adjust_for_multiplication:
            binary_array = [0] * (int_bits + fraction_bits) + binary_array
    else:
        positive_value = abs(number) - 1
        binary_string = bin(positive_value)[2:].zfill(
            int_bits + fraction_bits)  # Convert positive value to binary and fill leading zeros
        binary_array = [int(not int(bit)) for bit in binary_string]  # Invert bits for 2's complement
        # Add same number of bits of ones to the beginning of the array
        if adjust_for_multiplication:
            binary_array = [1] * (int_bits + fraction_bits) + binary_array
    return binary_array


def calculate_HH_neuron_model(dt=0.01, T=50.0, int_bits=8, fraction_bits=16, n_reset=1024, t_p_reset=100e-3, num_paths=1,
                              use_software_calculated_n_m_h=False):
    """
    Calculate and plot the result of a Hodgkin-Huxley neuron model
    :param dt: time step
    :param T: total time
    :param int_bits: number of integer bits for the crossbar (we use signed integer) (2's complement)
    :param fraction_bits: number of fraction bits for the crossbar
    :return: None
    """
    torch.set_default_dtype(torch.float64)
    HH_params = {
        'C_m': 1.0,
        'g_Na': 120,
        'g_K': 36,
        'g_L': 0.3,
        'V_Na': 115,
        'V_K': -12,
        'V_L': 10.613
    }
    """
    v_input = torch.tensor([
                    I,
                    n**4 * V,
                    n**4,
                    m**3 * h * V,
                    m**3 * h,
                    V,
                    1])
    """
    weights_for_v = torch.tensor([
        dt*(1 + dt) / (2*HH_params['C_m']),
        -HH_params['g_K'] * dt*(1 + dt) / (2*HH_params['C_m']),
        HH_params['g_K'] * HH_params['V_K'] * dt*(1 + dt) / (2*HH_params['C_m']),
        -HH_params['g_Na'] * dt*(1 + dt) / (2*HH_params['C_m']),
        HH_params['g_Na'] * HH_params['V_Na'] * dt*(1 + dt) / (2*HH_params['C_m']),
        1 + dt/2 - HH_params['g_L'] * dt*(1 + dt) / (2*HH_params['C_m']),
        HH_params['g_L'] * HH_params['V_L'] * dt*(1 + dt) / (2*HH_params['C_m']),
        ])
    binary_weights_for_v = torch.tensor([
        convert_to_binary_array(weight, int_bits, fraction_bits, adjust_for_multiplication=False)
        for weight in weights_for_v
    ], dtype=torch.float64)
    """
    n_input = torch.tensor([
                    n,
                    (10-V) * (1-n) / (exp((10 - V) / 10) - 1),
                    exp(-V / 80) * n,
                    torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
                ])
    """
    weights_for_n = torch.tensor([
        1 + dt/2,
        0.01 * dt * (1+dt) / 2,
        -0.125 * dt * (1+dt) / 2,
        dt * (1+dt) / 2
    ])
    binary_weights_for_n = torch.tensor([
        convert_to_binary_array(weight, int_bits, fraction_bits, adjust_for_multiplication=False)
        for weight in weights_for_n
    ], dtype=torch.float64)
    """
     m_input = torch.tensor([
                    m,
                    (25-V) * (1-m) / (exp((25 - V) / 10) - 1),
                    exp(-V / 18) * m,
                    torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
                ])
    """
    weights_for_m = torch.tensor([
        1 + dt/2,
        0.1 * dt * (1+dt) / 2,
        -4 * dt * (1+dt) / 2,
        dt * (1+dt) / 2
    ])
    binary_weights_for_m = torch.tensor([
        convert_to_binary_array(weight, int_bits, fraction_bits, adjust_for_multiplication=False)
        for weight in weights_for_m
    ], dtype=torch.float64)
    """
    h_input = torch.tensor([
                    h,
                    (1-h) * exp(-V / 20),
                    h / (exp((30 - V) / 10) + 1),
                    torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
                ])
    """
    weights_for_h = torch.tensor([
        1 + dt/2,
        0.07 * dt * (1+dt) / 2,
        -dt * (1+dt) / 2,
        dt * (1+dt) / 2
    ])
    binary_weights_for_h = torch.tensor([
        convert_to_binary_array(weight, int_bits, fraction_bits, adjust_for_multiplication=False)
        for weight in weights_for_h
    ], dtype=torch.float64)

    decoder = CurrentDecoder()

    v_crossbars, v_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
        binary_weights_for_v, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
    n_crossbars, n_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
        binary_weights_for_n, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
    m_crossbars, m_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
        binary_weights_for_m, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
    h_crossbars, h_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
        binary_weights_for_h, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)

    # binary_weights_for_v_left, binary_weights_for_v_right = torch.split(
    #     binary_weights_for_v, binary_weights_for_v.shape[1] // 2, dim=1)
    # binary_weights_for_n_left, binary_weights_for_n_right = torch.split(
    #     binary_weights_for_n, binary_weights_for_n.shape[1] // 2, dim=1)
    # binary_weights_for_m_left, binary_weights_for_m_right = torch.split(
    #     binary_weights_for_m, binary_weights_for_m.shape[1] // 2, dim=1)
    # binary_weights_for_h_left, binary_weights_for_h_right = torch.split(
    #     binary_weights_for_h, binary_weights_for_h.shape[1] // 2, dim=1)
    #
    # v_crossbar_left = build_binary_matrix_crossbar(
    #     binary_weights_for_v_left,
    #     n_reset=n_reset, t_p_reset=t_p_reset, set_voltage_difference=1.8)
    # v_crossbar_right = build_binary_matrix_crossbar(
    #     binary_weights_for_v_right,
    #     n_reset=n_reset, t_p_reset=t_p_reset, set_voltage_difference=1.8)
    #
    # n_crossbar_left = build_binary_matrix_crossbar(
    #     binary_weights_for_n_left,
    #     n_reset=n_reset, t_p_reset=t_p_reset, set_voltage_difference=1.8)
    # n_crossbar_right = build_binary_matrix_crossbar(
    #     binary_weights_for_n_right,
    #     n_reset=n_reset, t_p_reset=t_p_reset, set_voltage_difference=1.8)
    # m_crossbar_left = build_binary_matrix_crossbar(
    #     binary_weights_for_m_left,
    #     n_reset=n_reset, t_p_reset=t_p_reset, set_voltage_difference=1.8)
    # m_crossbar_right = build_binary_matrix_crossbar(
    #     binary_weights_for_m_right,
    #     n_reset=n_reset, t_p_reset=t_p_reset, set_voltage_difference=1.8)
    # h_crossbar_left = build_binary_matrix_crossbar(
    #     binary_weights_for_h_left,
    #     n_reset=n_reset, t_p_reset=t_p_reset, set_voltage_difference=1.8)
    # h_crossbar_right = build_binary_matrix_crossbar(
    #     binary_weights_for_h_right,
    #     n_reset=n_reset, t_p_reset=t_p_reset, set_voltage_difference=1.8)

    # v_crossbar = build_binary_matrix_crossbar(binary_weights_for_v, n_reset=n_reset, t_p_reset=t_p_reset,
    #                                  set_voltage_difference=1.8)
    # n_crossbar = build_binary_matrix_crossbar(binary_weights_for_n, n_reset=n_reset, t_p_reset=t_p_reset,
    #                                     set_voltage_difference=1.8)
    # m_crossbar = build_binary_matrix_crossbar(binary_weights_for_m, n_reset=n_reset, t_p_reset=t_p_reset,
    #                                     set_voltage_difference=1.8)
    # h_crossbar = build_binary_matrix_crossbar(binary_weights_for_h, n_reset=n_reset, t_p_reset=t_p_reset,
    #                                     set_voltage_difference=1.8)

    # v_left_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(v_crossbar_left, binary_weights_for_v)
    # v_right_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(v_crossbar_right, binary_weights_for_v)
    # n_left_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(n_crossbar_left, binary_weights_for_n)
    # n_right_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(n_crossbar_right, binary_weights_for_n)
    # m_left_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(m_crossbar_left, binary_weights_for_m)
    # m_right_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(m_crossbar_right, binary_weights_for_m)
    # h_left_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(h_crossbar_left, binary_weights_for_h)
    # h_right_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(h_crossbar_right, binary_weights_for_h)

    # v_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(v_crossbar, binary_weights_for_v)
    # n_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(n_crossbar, binary_weights_for_n)
    # m_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(m_crossbar, binary_weights_for_m)
    # h_bit_line_possible_outputs = decoder.calibrate_binary_crossbar_output_current_thresholds(h_crossbar, binary_weights_for_h)

    master_t_list = []
    master_V_list = []
    master_n_list = []
    master_m_list = []
    master_h_list = []

    for _ in range(num_paths):
        V = 0
        n = 0.0
        m = 0
        h = 0
        I = 10

        V_result = 0
        n_result = 0
        m_result = 0
        h_result = 0

        t = 0
        t_list = []
        V_list = []
        n_list = []
        m_list = []
        h_list = []

        theoretical_v_list = []
        theoretical_n_list = []
        theoretical_m_list = []
        theoretical_h_list = []

        theoretical_V = V
        theoretical_n = n
        theoretical_m = m
        theoretical_h = h

        # pause at time = t_to_pause
        t_to_pause_start = 0.75
        t_to_pause_end = 0.85

        step_count = 0
        while t < T:
            if t_to_pause_end > t > t_to_pause_start:
                print("paused")


            # try to reprogram every 50 time steps
            step_count += 1
            # if step_count % 50 == 0:
            #     v_crossbars, v_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
            #         binary_weights_for_v, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
            #     n_crossbars, n_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
            #         binary_weights_for_n, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
            #     m_crossbars, m_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
            #         binary_weights_for_m, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
            #     h_crossbars, h_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
            #         binary_weights_for_h, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)

            random_number_std_dev = 0.1
            t_list.append(t)
            V_list.append(V)
            n_list.append(n)
            m_list.append(m)
            h_list.append(h)

            theoretical_v_list.append(theoretical_V)
            theoretical_n_list.append(theoretical_n)
            theoretical_m_list.append(theoretical_m)
            theoretical_h_list.append(theoretical_h)

            # How many times to repeat the simulation for each time step and find avg
            num_repeats = 1

            temp_sum_v = 0
            temp_sum_n = 0
            temp_sum_m = 0
            temp_sum_h = 0

            repeat_count = 0
            while repeat_count < num_repeats:
                repeat_count += 1
                v_input = torch.tensor([
                    I,
                    n**4 * V,
                    n**4,
                    m**3 * h * V,
                    m**3 * h,
                    V,
                    1
                ])
                n_input = torch.tensor([
                    n,
                    (10-V) * (1-n) / (exp((10 - V) / 10) - 1),
                    exp(-V / 80) * n,
                    torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
                ])
                m_input = torch.tensor([
                    m,
                    (25-V) * (1-m) / (exp((25 - V) / 10) - 1),
                    exp(-V / 18) * m,
                    torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
                ])
                h_input = torch.tensor([
                    h,
                    (1-h) * exp(-V / 20),
                    h / (exp((30 - V) / 10) + 1),
                    torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
                ])

                # expected_V is dot product of v_input and v_weights
                # RuntimeError: dot : expected both vectors to have same dtype, but found Long and Double
                expected_V = torch.dot(v_input.double(), weights_for_v.double()).item()
                expected_n = torch.dot(n_input.double(), weights_for_n.double()).item()
                expected_m = torch.dot(m_input.double(), weights_for_m.double()).item()
                expected_h = torch.dot(h_input.double(), weights_for_h.double()).item()

                # clamp all input to -2**int_bits, 2**int_bits - 1
                v_input = torch.clamp(v_input, -2**(int_bits-1), 2**(int_bits-1) - 1)
                n_input = torch.clamp(n_input, -2**(int_bits-1), 2**(int_bits-1) - 1)
                m_input = torch.clamp(m_input, -2**(int_bits-1), 2**(int_bits-1) - 1)
                h_input = torch.clamp(h_input, -2**(int_bits-1), 2**(int_bits-1) - 1)

                v_binary_input = torch.tensor([
                    convert_to_binary_array(weight, int_bits, fraction_bits, adjust_for_multiplication=False)
                    for weight in v_input
                ])
                n_binary_input = torch.tensor([
                    convert_to_binary_array(weight, int_bits, fraction_bits, adjust_for_multiplication=False)
                    for weight in n_input
                ])
                m_binary_input = torch.tensor([
                    convert_to_binary_array(weight, int_bits, fraction_bits, adjust_for_multiplication=False)
                    for weight in m_input
                ])
                h_binary_input = torch.tensor([
                    convert_to_binary_array(weight, int_bits, fraction_bits, adjust_for_multiplication=False)
                    for weight in h_input
                ])

                # correct code to use
                """correct"""
                # now included /2**(2*frac bits)
                V_result = calculate_vmm_result_split_into_subsection(v_crossbars, v_crossbars_possible_outputs, decoder, v_binary_input) / (2 ** (2 * fraction_bits))  # , debug_weight_matrix=binary_weights_for_v
                # V_result = expected_V  # ONLY FOR FULL SOFTWARE COMPUTATION
                # V_result = binary_expected_V  # ONLY FOR FULL SOFTWARE COMPUTATION
                if use_software_calculated_n_m_h:
                    n_result = expected_n
                    m_result = expected_m
                    h_result = expected_h
                else:
                    n_result = calculate_vmm_result_split_into_subsection(n_crossbars, n_crossbars_possible_outputs, decoder, n_binary_input) / (2 ** (2 * fraction_bits))
                    m_result = calculate_vmm_result_split_into_subsection(m_crossbars, m_crossbars_possible_outputs, decoder, m_binary_input) / (2 ** (2 * fraction_bits))
                    h_result = calculate_vmm_result_split_into_subsection(h_crossbars, h_crossbars_possible_outputs, decoder, h_binary_input) / (2 ** (2 * fraction_bits))

                # V_result = V_result / (2 ** (2 * fraction_bits))
                # n_result = n_result / (2 ** (2 * fraction_bits))
                # m_result = m_result / (2 ** (2 * fraction_bits))
                # h_result = h_result / (2 ** (2 * fraction_bits))

                if V_result > 2 ** (int_bits - 1) - 1:
                    V_result = 2 ** (int_bits - 1) - 1
                if V_result < -2 ** (int_bits - 1):
                    V_result = -2 ** (int_bits - 1)
                if n_result > 1:
                    n_result = 1
                if m_result > 1:
                    m_result = 1
                if h_result > 1:
                    h_result = 1
                if n_result < 0:
                    n_result = 0
                if m_result < 0:
                    m_result = 0
                if h_result < 0:
                    h_result = 0

                # if abs(n_result - n) > 2 * 1.3 * dt:
                #     print("pause, n changed too much")
                #     n_crossbars, n_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
                #         binary_weights_for_n, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
                #     repeat_count -= 1
                #     continue
                #
                # if abs(m_result - m) > 2 * 11 * dt:
                #     print("pause, m changed too much")
                #     m_crossbars, m_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
                #         binary_weights_for_m, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
                #     repeat_count -= 1
                #     continue
                #
                # if abs(h_result - h) > 2 * 1.1 * dt:
                #     print("pause, h changed too much")
                #     h_crossbars, h_crossbars_possible_outputs = build_binary_matrix_crossbar_split_into_subsections(
                #         binary_weights_for_h, num_row_splits=2, num_col_splits=8, n_reset=n_reset, t_p_reset=t_p_reset)
                #     repeat_count -= 1
                #     continue

                temp_sum_v += V_result
                temp_sum_n += n_result
                temp_sum_m += m_result
                temp_sum_h += h_result

            V = temp_sum_v / num_repeats
            n = temp_sum_n / num_repeats
            m = temp_sum_m / num_repeats
            h = temp_sum_h / num_repeats

            # Theoretical, start from scratch ignoring previous values
            theoretical_v_input = torch.tensor([
                I,
                theoretical_n ** 4 * theoretical_V,
                theoretical_n ** 4,
                theoretical_m ** 3 * theoretical_h * theoretical_V,
                theoretical_m ** 3 * theoretical_h,
                theoretical_V,
                1
            ])
            theoretical_n_input = torch.tensor([
                1 / (exp((10 - theoretical_V) / 10) - 1),
                theoretical_V / (exp((10 - theoretical_V) / 10) - 1),
                theoretical_n / (exp((10 - theoretical_V) / 10) - 1),
                theoretical_n * theoretical_V / (exp((10 - theoretical_V) / 10) - 1),
                exp(-theoretical_V / 80) * theoretical_n,
                theoretical_n,
                torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
            ])
            theoretical_m_input = torch.tensor([
                1 / (exp((25 - theoretical_V) / 10) - 1),
                theoretical_V / (exp((25 - theoretical_V) / 10) - 1),
                theoretical_m / (exp((25 - theoretical_V) / 10) - 1),
                theoretical_m * theoretical_V / (exp((25 - theoretical_V) / 10) - 1),
                exp(-theoretical_V / 18) * theoretical_m,
                theoretical_m,
                torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
            ])
            theoretical_h_input = torch.tensor([
                exp(-theoretical_V / 20),
                theoretical_h * exp(-theoretical_V / 20),
                theoretical_h / (exp((30 - theoretical_V) / 10) + 1),
                theoretical_h,
                torch.normal(0.0, random_number_std_dev, (1,)).item()  # noise
            ])

            theoretical_V = torch.clamp(torch.dot(theoretical_v_input.double(), weights_for_v.double()), -2 ** (int_bits-1), 2 ** (int_bits-1) - 1).item()
            theoretical_n = torch.clamp(torch.dot(theoretical_n_input.double(), weights_for_n.double()), 0.0, 1.0).item()
            theoretical_m = torch.clamp(torch.dot(theoretical_m_input.double(), weights_for_m.double()), 0.0, 1.0).item()
            theoretical_h = torch.clamp(torch.dot(theoretical_h_input.double(), weights_for_h.double()), 0.0, 1.0).item()

            t += dt
            print(t)
            # print("\t", V * 2**fraction_bits, expected_V * 2**fraction_bits, theoretical_V, V * 2 ** fraction_bits - expected_V * 2 ** fraction_bits)
            print("\t", V, expected_V, theoretical_V)
            print("\t", n, expected_n, theoretical_n)
            print("\t", m, expected_m, theoretical_m)
            print("\t", h, expected_h, theoretical_h)
            print("\t", I)
        master_t_list.append(t_list)
        master_V_list.append(V_list)
        master_n_list.append(n_list)
        master_m_list.append(m_list)
        master_h_list.append(h_list)
        # Plot the results, in 4 subplots
        plt.subplot(2, 2, 1)
        plt.title("n")
        plt.plot(t_list, n_list, label="n")
        plt.subplot(2, 2, 2)
        plt.title("m")
        plt.plot(t_list, m_list, label="m")
        plt.subplot(2, 2, 3)
        plt.title("h")
        plt.plot(t_list, h_list, label="h")
        plt.subplot(2, 2, 4)
        plt.title("V")
        plt.plot(t_list, V_list, label="V")
        plt.show()

    # plot a light thin line for each t_list in master_t_list
    # plot a thick line for the average of each index across all t_lists in master_V_list

    plt.subplot(2, 2, 1)
    plt.title("n")
    t_list = master_t_list[0]
    for n_list in master_n_list:
        # make the line thin and light
        plt.plot(t_list, n_list, linewidth=0.5, alpha=0.5, label="n")
    # plot average
    plt.plot(t_list, np.mean(master_n_list, axis=0), linewidth=2, alpha=1, label="n")
    plt.plot(t_list, theoretical_n_list, linewidth=1, alpha=1)
    plt.subplot(2, 2, 2)
    plt.title("m")
    for m_list in master_m_list:
        plt.plot(t_list, m_list, linewidth=0.5, alpha=0.5, label="m")
    # plot average
    plt.plot(t_list, np.mean(master_m_list, axis=0), linewidth=2, alpha=1, label="m")
    plt.plot(t_list, theoretical_m_list, linewidth=1, alpha=1)
    plt.subplot(2, 2, 3)
    plt.title("h")
    for h_list in master_h_list:
        plt.plot(t_list, h_list, linewidth=0.5, alpha=0.5, label="h")
    # plot average
    plt.plot(t_list, np.mean(master_h_list, axis=0), linewidth=2, alpha=1, label="h")
    plt.plot(t_list, theoretical_h_list, linewidth=1, alpha=1)
    plt.subplot(2, 2, 4)
    plt.title("V")
    for V_list in master_V_list:
        plt.plot(t_list, V_list, linewidth=0.5, alpha=0.5, label="V")
    # plot average
    plt.plot(t_list, np.mean(master_V_list, axis=0), linewidth=2, alpha=1, label="V")
    plt.plot(t_list, theoretical_v_list, linewidth=1, alpha=1)
    plt.show()
    # save all numbers to a file
    import datetime
    current_time = datetime.datetime.now()
    for i in range(num_paths):
        with open(f"data_{current_time}_{i}.txt", "w") as f:
            f.write(f"t_list: {master_t_list[i]}\n")
            f.write(f"V_list: {master_V_list[i]}\n")
            f.write(f"n_list: {master_n_list[i]}\n")
            f.write(f"m_list: {master_m_list[i]}\n")
            f.write(f"h_list: {master_h_list[i]}\n")




    # # Plot the results, in 4 subplots
    # plt.subplot(2, 2, 1)
    # plt.title("n")
    # plt.plot(t_list, n_list, label="n")
    # plt.subplot(2, 2, 2)
    # plt.title("m")
    # plt.plot(t_list, m_list, label="m")
    # plt.subplot(2, 2, 3)
    # plt.title("h")
    # plt.plot(t_list, h_list, label="h")
    # plt.subplot(2, 2, 4)
    # plt.title("V")
    # plt.plot(t_list, V_list, label="V")
    # plt.show()








"""
TODO:
1. fix the produce crossbar function - don't do the fill by all 1s or 0s
2. in the compute function, do something like, first chunk of crossbar output should become subtracted (2's complement)
3. in the compute function, after summing chunks for each input bit slice, make sure to subtract the MSB's 2's complement 
"""

def main():
    # test_sequential_bit_input_inference_and_power()
    calculate_HH_neuron_model(dt=0.01, n_reset=0, T=15, fraction_bits=50, t_p_reset=100, num_paths=1,
                              use_software_calculated_n_m_h=False)

if __name__ == "__main__":
    main()
