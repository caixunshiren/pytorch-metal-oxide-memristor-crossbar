
# PyTorch Metal-Oxide Memristor Crossbar

### Overview

PyTorch implementation of the paper ["Comprehensive Compact Phenomenological Modeling of Integrated Metal-Oxide Memristors"][1]  
using the crossbar line resistance model derived by [[2]][2].
The baseline of the work is to simulate a realistic memristor crossbar array that we can treat as a real device.

The layout/roadmap of the package is as follows:

```
Metal-oxide crossbar ---> VMM Engine ---> PyTorch Model
```

**[Metal-Oxide Crossbar]** 

The crossbar is an exact circuit simulation of a memristor crossbar, with current and 
voltage flowing through the devices. The memristor elements are implemented with the model fitted on [real devices][1].
They will change conductance accordingly based on programming pulses presented at its terminals. Word lines, bit lines, 
and sensing circuits are implemented. During inference, memristor elements are treated as resistors, and voltage at each
point is calculated using a [line resistance model][2].

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

The VMM engine utilizes the crossbar to perform vector matrix multiplication. It encodes matrices and vectors to program
as conductances and feed in as voltage pulses in the analog crossbar circuit. There are many implementations, and one 
can implement those schemes in literatures. Different schemes trade of math fidelity, circuit complexity, circuit 
components (footprint), power consumptions, and clock cycles.

The following engines are implemented:

- TODO: fill in this part

**[PyTorch Model]**

The PyTorch model is a wrapper around the VMM engine to make it a PyTorch Module. While supporting full circuit-level 
simulation, running the high fidelity model is time-consuming. Therefore, the PyTorch model provides an approximation to
work around the issue. 

TODO: fill in this part, talk about the approximation scheme used

### Installation

```
pip install -r requirements.txt
```

### Usage

`main.py` provides some example functions to play around the crossbar.

[1]: https://ieeexplore.ieee.org/abstract/document/9047174
[2]: https://ieeexplore.ieee.org/document/6473873
