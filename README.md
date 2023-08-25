
# PyTorch Metal-Oxide Memristor Crossbar

### Overview

PyTorch implementation of the paper ["Comprehensive Compact Phenomenological Modeling of Integrated Metal-Oxide Memristors"][1] using the crossbar line resistance model derived by [[2]][2]. The baseline of the work is to simulate a realistic memristor crossbar array that we can treat as a real device.

The layout/roadmap of the package is as follows:

```
Metal-oxide crossbar ---> VMM Engine ---> PyTorch Model
```

**[Metal-Oxide Crossbar]** 

The crossbar is an exact circuit simulation of a memristor crossbar, with current and voltage flowing through the devices. 
The memristor elements are implemented with the model fitted on [real devices][1]. 
They will change conductance accordingly based on programming pulses presented at its terminals. 
Word lines, bit lines, and sensing circuits are implemented. 
During inference, memristor elements are treated as resistors, and voltage at each point is calculated using a [line resistance model][2].

As a result, the following **non-idealities** are captured:

_[Memristor model wise]_
- Nonlinear I-V curve
- Nonlinear conductance change
- Set/Reset asymmetry, stuck devices
- Thermal effect

_[Crossbar wise]_
- Sneak path current
- Voltage drop due to line resistance

**[VMM Engine]** 

The VMM engine utilizes the crossbar to perform vector matrix multiplication. 
It encodes matrices and vectors to program as conductances and feed in as voltage pulses in the analog crossbar circuit. 
There are many implementations, and one can implement those schemes in literatures. 
Different schemes trade of math fidelity, circuit complexity, circuit components (footprint), power consumptions, and clock cycles.

The following engines are implemented:

- TODO: fill in this part

**[PyTorch Model]**

The PyTorch model is a wrapper around the VMM engine to make it a PyTorch Module. 
While supporting full circuit-level simulation, running the high fidelity model is time-consuming. 
Therefore, the PyTorch model provides an approximation to work around the issue. 

TODO: fill in this part, talk about the approximation scheme used

### Installation

```
pip install -r requirements.txt
```

### Usage

`main.py` provides some example functions to play around the crossbar.

#### **Memristor**

A memristor is a resistive memory device that is programmable. 
We implemented the memristor according to the model in [[1]][1].
The current implementation has 4 types of memristors, and a single memristor element can be imported as follows:

```
from memristor.devices import StaticMemristor, DynamicMemristor, DynamicMemristorFreeRange, DynamicMemristorStuck
```

DynamicMemristor is a subclass of StaticMemristor, and DynamicMemristorFreeRange and DynamicMemristorStuck are subclasses of DynamicMemristor.
Static memrisotr supports inference only, and all dynamic memristors support both inference and programming. 

The following parameters define a memristor:

```
# user set parameters
self.g_0 = g_0  # initial, ideal conductance in S. Typically, 3.16-316 uS is used.
self.t = None   # operating temperature
self.f = None   # operating frequency

# useful parameters
self.g_linfit = None  # best fitted linear conductance to approximate inference behavior
                      # this is recomputed every time calibrate() is called

# below are parameters calibrated from the above parameters
# these parameter changes with respect to temperature and frequency, so recalibration is needed when operating environment changes
self.u_A1 = None
self.u_A3 = None
self.sigma_A1 = None
self.sigma_A3 = None
self.g_linfit = None  # best fitted conductance
self.d2d_var = np.random.normal(0, 1, 1).item()
```

The following are the functions supported in a static memristor:

```
__init__(self, g_0): initialize a static memristor, given the initial conductance
calibrate(self, t, f): calibrate the memristor at a given temperature and frequency, this must be done before running inference
noise_free_dc_iv_curve(self, v): given a voltage, calculate the current with non-linearity only (noise free)
inference(self, v): given a voltage, calculate the current with non-linearity and non-idealities included
```

Dynamic memristor supports two additional functions:

```
set(self, V_p, t_p): given a programming voltage and time, set the memristor (increase conductance)
reset(self, V_p, t_p): given a programming voltage and time, reset the memristor (decrease conductance)
```

**Table 1: Typical Ranges of Parameters:**

| Parameter | Type                      | Typical Range   | Code                  |
|-----------|---------------------------|-----------------|-----------------------|
| `g_0`     | Initial conductance       | 3.16 - 316 uS   | `3.16e-6` - `3.16e-4` |
| `t`       | Operating Temperature     | 300 - 400 K     | `300` - `375`         |
| `f`       | Operating Frequency       | 1 - 1000 MHz    | `1e6` - `1e9`         |
| `v`       | Inference Voltage         | -0.4 - 0.4 V    | `-0.4` - `0.4`        |
| `V_p`     | Programming Voltage (SET) | -0.8 - -1.5 V   | `-0.8` - `-1.5`       |
| `V_p`     | Programming Voltage (RESET) | 0.8 - 1.15 V    | `0.8` - `1.15`        |
| `t_p`     | Programming Time          | 100 ns - 100 ms | `1e-7` - `1e-1`       |

`DynamicMemristorFreeRange` and `DynamicMemristorStuck`:

When programming the conductance towards the range limit, the DynamicMemristor will stop updating if it reaches the extreme.
This is because the model is only accurate for interpolation between the 3.16 - 316 uS range. 
We support two additional models depending on the desired simulation behavior.

`DynamicMemristorFreeRange` allows the memristor to go beyond the range limit while fidelity is not guaranteed.

`DynamicMemristorStuck` stuck the memristor at the extreme to simulate stuck devices.

**Example Code:**

Inference using a StaticMemristor:

```
from memristor.devices import StaticMemristor

frequency = 1e8  # hz
temperature = 60+273  # Kelvin
g_0 = 50e-6  # S
v = 0.3  # V
memristor = StaticMemristor(g_0)
memristor.calibrate(temperature, frequency)
print("ideal naive linear estimate:", v * g_0)
print("ideal naive non-linear estimate:", memristor.noise_free_dc_iv_curve(v))
print("actual device inference:", memristor.inference(v))
```

Programming using a DynamicMemristor:

```
from memristor.devices import DynamicMemristor

v_p = 1.0 # range [−0.8 V to −1.5 V]/[0.8 V to 1.15 V] RESET/SET
t_p = 0.5e-3 # programming pulse duration
g_0 = 60e-6
frequency = 1e8  # hz
temperature = 273 + 60  # Kelvin
OPERATION = "SET"  # "SET" or "RESET"
iterations = 10

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
```

Example I-V/Device programming Plots: 

Checkout `Notebooks/single device sim.ipynb`.

[1]: https://ieeexplore.ieee.org/abstract/document/9047174
[2]: https://ieeexplore.ieee.org/document/6473873
