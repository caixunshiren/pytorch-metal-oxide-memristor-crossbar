
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

#### 1. **Memristor**

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


#### 2. **Crossbar**

A crossbar is a 2D array of memristors connected by wordlines and bitlines. 
VMM can be performed using Kirchoff's law and Ohm's law. 
Due to existence of line resistance, the real VMM circuit is more complicated than a simple matrix multiplication.

The crossbar module implements a 0T1R passive crossbar that supports inference and programming. 
A resistive [crossbar model][2] is used to calculate voltage at each point of the device.

To use a crossbar, import from

```
from memristor.crossbar.model import LineResistanceCrossbar
```

The following parameters define a crossbar:

```
memristor_model: memristor model class
memristor_params: dictionary of the memristor model param, this depends on the memristor model
ideal_w: (n,m) numpy/torch matrix of ideal conductances be programed. Size of crossbar depends on this matrix
crossbar_params: dictionary of crossbar parameters
```

See `memristor.crossbar.model.initialize_memristor` to see how memristor parameters are used or implement your own 
`initialize_memristor` for your custom memristor model. 
Table 2 shows all the `crossbar_params` and their recommended values.

**Table 2: Crossbar Parameters:**

| Name            | Desription                        | Typical Value                | Code        |
|-----------------|-----------------------------------|------------------------------|-------------|
| `r_wl`          | Wordline resistance               | 20 ohm                       | `20`        |
| `r_bl`          | Bitline resistance                | 20 ohm                       | `20`        |
| `r_in`          | Input sensing circuit resistance  | 10 ohm                       | `20`        |
| `r_out`         | Output Sensing circuit resistance | 10 ohm                       | `20`        |
| `V_SOURCE_MODE` | Discussed below                   | Single/double/3 quater sided | shown below |

`V_SOURCE_MODE` defines how the input voltage is applied to the crossbar. 
It can be either applied one side of bitline and wordline, both sides of bitline and wordline, or 2 sides for wordline and 1 side for bitline.
Depending on the application, applying from both sides can mitigate effect of line resistance, while increasing the power consumption and circuit complexity.

`V_SOURCE_MODE` can be set to one of the following:

```
V_SOURCE_MODE = "SINGLE_SIDE" or "|_"  # single sided
V_SOURCE_MODE = "DOUBLE_SIDE" or "|=|"  # double sided
V_SOURCE_MODE = "THREE_QUATER_SIDE" or "|_|"  # 3 quarter sided
```

We demonstrate the crossbar functions with a toy example. Below is an example of a crossbar with 16x16 memristors.

To initialize a crossbar:
```
torch.set_default_dtype(torch.float64)

crossbar_params = {'r_wl': 20, 'r_bl': 20, 'r_in':10, 'r_out':10, 'V_SOURCE_MODE':'|_|'}
memristor_model = StaticMemristor
memristor_params = {'frequency': 1e8, 'temperature': 273 + 40}
ideal_w = torch.FloatTensor(16, 16).uniform_(10, 300).double()*1e-6

crossbar = LineResistanceCrossbar(memristor_model, memristor_params, ideal_w, crossbar_params)
```

To perform VMM:
```
v_wl_applied = torch.FloatTensor(16,).uniform_(0, 0.4).double()
v_bl_applied = torch.zeros(16, ).double()
x = crossbar.lineres_memristive_vmm(v_wl_applied, v_bl_applied, order=1)
```

_***About the `order` parameter:_ 

The way we calculate the voltage at each point of the crossbar is by solving a linear system of equations.
Therefore, we assume that each memristor is a perfect linear resistor, and we use its `g_linfit` to calculate the voltage.
If we increase the order to 2, then we will run inference with the calculated voltage and recalculate a better `g_linfit`.
This process can be repeated for `order` number of times to converge to a better simulation fidelity. 
For default cases, `order=1` is sufficient.

To program the crossbar:
```
n_reset = 40
t_p_reset = 0.5e-3
v_p_bl = 1.5 * torch.cat([torch.linspace(1, 1.2, 16)], dim=0)
for j in tqdm(range(n_reset)):
    crossbar.lineres_memristive_programming(torch.zeros(16, ), v_p_bl, t_p_reset, threshold=0.8)
```

The above example applies reset to all the memristors using `crossbar.lineres_memristive_programming`. 
Performing set and reset depends on the voltage difference at each terminal of the memristors. 
In this case, since the wordline voltage is 0 and bitline voltage is non-zero, all the memristors will be reset.
Only voltage difference with magnitude greater than 0.8 V will have the memristor be programmed.
By default, `threshold=0.8` for `crossbar.lineres_memristive_programming`.
To perform programming of a single memristor, we can the a row by row or column by column scheme where we make sure
all other memristors have voltage differences below the threshold. 

**Example Code:**

Test Inference:

```
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

```

Power consumption tracking is supported. See Test Power:

```
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
```



[1]: https://ieeexplore.ieee.org/abstract/document/9047174
[2]: https://ieeexplore.ieee.org/document/6473873
