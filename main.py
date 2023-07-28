import matplotlib.pyplot as plt

from memristor.devices import StaticMemristor, DynamicMemristor, DynamicMemristorFreeRange, DynamicMemristorStuck
from memristor.crossbar.model import LineResistanceCrossbar
import torch
import numpy as np
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
        m, n = crossbar.m, crossbar.n
        # crossbar_bit_lines: {bit_line_index: {possible_value: [sum_current_over_iterations, number_of_occurrence]}}
        crossbar_bit_lines = {i: {} for i in range(n)}
        # Create all possible input combinations
        X = torch.zeros(size=m)
        for i in range(itr):
            output = torch.vmm(binary_crossbar, X.unsqueeze(-1)).squeeze() # [n, ]
            output_current = crossbar.lineres_memristive_vmm(0.4 * X, torch.zeros([n]))
            for j in range(n):
                if output[j] not in crossbar_bit_lines[j]:
                    crossbar_bit_lines[j][output[j]] = [output_current, 1]
                else:
                    crossbar_bit_lines[j][output[j]][0] += output_current
                    crossbar_bit_lines[j][output[j]][1] += 1
            while X.sum() < m:
                index = m - 1
                while X[index] == 1:
                    X[index] = 0
                    index -= 1
                X[index] = 1
                output = torch.vmm(binary_crossbar, X.unsqueeze(-1)).squeeze()  # [n, ]
                output_current = crossbar.lineres_memristive_vmm(0.4 * X, torch.zeros([n]))
                for j in range(n):
                    if output[j] not in crossbar_bit_lines[j]:
                        crossbar_bit_lines[j][output[j]] = [output_current, 1]
                    else:
                        crossbar_bit_lines[j][output[j]][0] += output_current
                        crossbar_bit_lines[j][output[j]][1] += 1
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
            for possible_output in crossbar_bit_lines[j]:
                diff = abs(crossbar_bit_lines[j][possible_output] - output_current[j])
                if diff < min_diff:
                    min_diff = diff
                    output[j] = possible_output
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


def test_sequential_bit_input_inference_and_power():
    torch.set_default_dtype(torch.float64)
    weights = torch.tensor([
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1]
    ])
    # crossbar = build_binary_matrix_crossbar(torch.ones_like(weights), n_reset=64, t_p_reset=0.5e-3)
    torch.set_default_dtype(torch.float64)
    crossbar_params = {'r_wl': 20, 'r_bl': 20, 'r_in': 10, 'r_out': 10, 'V_SOURCE_MODE': '|_|'}
    memristor_model = DynamicMemristorStuck
    memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
    m, n = weights.shape
    ideal_w = torch.ones([n, m]) * 65e-6  # shape is [number_of_cols, number_of_rows]
    crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)

    decoder = CurrentDecoder()
    max_current = decoder.calibrate_max_current(crossbar)

    torch.set_default_dtype(torch.float64)
    crossbar_params = {'r_wl': 20, 'r_bl': 20, 'r_in': 10, 'r_out': 10, 'V_SOURCE_MODE': '|_|'}
    memristor_model = DynamicMemristorStuck
    memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
    m, n = weights.shape
    ideal_w = torch.ones([n, m]) * 65e-6  # shape is [number_of_cols, number_of_rows]
    crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)
    min_current = decoder.calibrate_min_current(crossbar)

    crossbar = build_binary_matrix_crossbar(weights)

    input_vector = torch.tensor([
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1]
    ])
    final_result = 0
    for bit in range(8):
        v_wl_applied = input_vector[:, bit] * 0.4
        v_bl_applied = torch.zeros(8)
        print("input", v_wl_applied)
        x = crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied, log_power=True)
        # shift and add
        decoded = decoder.apply_2_bits(x, min_current, max_current)
        bit_result = 0
        for i, decoded_bit in enumerate(decoded):
            bit_result += decoded_bit * 2 ** (8 - i - 1)
        print("bit result", bit_result)
        final_result += bit_result * 2 ** (8 - bit - 1)
    print("final result", final_result)


def main():
    test_sequential_bit_input_inference_and_power()

if __name__ == "__main__":
    main()
