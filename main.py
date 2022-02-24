from memristor.devices import StaticMemristor


def main():
    frequency = 10e8  # hz
    temperature = 40  # Celcius
    g_0 = 50e-6  # S
    v = 0.4  # V
    memristor = StaticMemristor(g_0)
    memristor.calibrate(temperature, frequency)
    for i in range(10):
        print(memristor.inference(v))
    ideal_i = v * g_0
    print("ideal naive estimate:", ideal_i)


if __name__ == "__main__":
    main()
