from memristor.devices import StaticMemristor, DynamicMemristor


def graph_I_V(n, v_range, g_0, frequency, temperature):
    import matplotlib.pyplot as plt
    import numpy as np
    for j in range(n):
        memristor = StaticMemristor(g_0)
        memristor.calibrate(temperature, frequency)
        I = [memristor.inference(v) for v in np.linspace(v_range[0], v_range[1], 50)]
        I_linfit = [v * memristor.g_linfit for v in np.linspace(v_range[0], v_range[1], 50)]
        plt.plot(np.linspace(v_range[0], v_range[1], 50), I, label=f"simulation {j}")
        plt.plot(np.linspace(v_range[0], v_range[1], 50), I_linfit, label=f"simulation {j} best fit")
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
        memristor = DynamicMemristor(g_0)
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

def main():
    frequency = 1e8  # hz
    temperature = 273 + 60  # Kelvin
    g_0 = 50e-6  # S
    v = 0.3  # V
    memristor = StaticMemristor(g_0)
    memristor.calibrate(temperature, frequency)
    for i in range(10):
        print(memristor.inference(v))
    ideal_i = v * g_0
    print("ideal naive linear estimate:", ideal_i)
    print("ideal naive non-linear estimate:", memristor.noise_free_dc_iv_curve(v))

    graph_I_V(2, [-0.4, 0.4], g_0, frequency, temperature)

    v_p = 1.0 # range [−0.8 V to −1.5 V]/[0.8 V to 1.15 V]
    t_p = 0.5e-3 # programming pulse duration
    g_0 = 65e-6
    frequency = 1e8  # hz
    temperature = 273 + 60  # Kelvin
    #plot_conductance_multiple(100, 200, g_0, t_p, v_p, temperature, frequency, OPERATION="SET")

if __name__ == "__main__":
    main()
