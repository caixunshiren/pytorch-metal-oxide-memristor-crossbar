
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
- Nonlinear conductance change
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

**[Memristor]**
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
self.g_0 = g_0  # initial, ideal conductance in S. Typically, 1-100 uS is used.
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
inference(self, v): given a voltage, calculate the current
```

Dynamic memristor supports two additional functions:

```
set(self, V_p, t_p): given a programming voltage and time, set the memristor (increase conductance)
reset(self, V_p, t_p): given a programming voltage and time, reset the memristor (decrease conductance)
```




[1]: https://ieeexplore.ieee.org/abstract/document/9047174
[2]: https://ieeexplore.ieee.org/document/6473873
