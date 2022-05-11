from memristor.devices import StaticMemristor, DynamicMemristor


def graph_I_V(v_range, g_0, frequency, temperature):
    import matplotlib.pyplot as plt
    import numpy as np
    memristor = StaticMemristor(g_0)
    memristor.calibrate(temperature, frequency)
    I = [memristor.inference(v) for v in np.linspace(v_range[0], v_range[1], 50)]
    I_ohms_law = [v * g_0 for v in np.linspace(v_range[0], v_range[1], 50)]
    plt.plot(np.linspace(v_range[0], v_range[1], 50), I, label="simulation")
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

def main():
    frequency = 10e8  # hz
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

    #graph_I_V([-0.4, 0.4], g_0, frequency, temperature)

    v_p = 1.1 # range [−0.8 V to −1.5 V]/[0.8 V to 1.15 V]
    t_p = 0.5*10e-3 # programming pulse duration
    g_0 = 65e-6
    frequency = 10e8  # hz
    temperature = 273 + 60  # Kelvin
    plot_conductance(80, g_0, t_p, v_p, temperature, frequency, OPERATION="SET")

if __name__ == "__main__":
    main()
