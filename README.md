# FGYM
Demonstrating the usage of FGYM: A Toolkit for benchmarking FPGA-accelerated Reinforcement Learning


## Overview

### Introduction

FPGA-based heterogeneous computing platforms are promising candidates to enable fast training of Reinforcement Learning (RL) agents. Typically, an RL agent for an environment is trained via interactions with a software that simulates the environment. While several toolkits exist to quickly deploy RL training on CPU or GPU, there lacks a similar toolkit for FPGAs. 
![High-Level-ovvw](https://github.com/CatherineMeng/FGYM-user-demo/blob/main/fig/ovvw.png)

Interfacing GYM with an FPGA based RL agent remains a manual and time-consuming process. To address this issue and ease the deployment process of RL using FPGAs, we demonstrate FGYM (FPGA-GYM) - a toolkit that generates an end-to-end interface between the simulation environments running on the CPU and agents running on the FPGA. FGYM supports all GYM environments specified in the python host program and automatically generates the on-chip memory interface using PCIe. FGYM supports multiple levels of parallelism including vectorized agent-environment interactions and memory port aggregation. Building upon [VITIS](https://github.com/Xilinx/Vitis-Tutorials) execution flow, It also provides post-execution profiling results for users to identify the execution bottlenecks. 

By bridging python host code with FPGA kernel through pyopencl library and [XRT](https://github.com/Xilinx/XRT), FGYM targets developers and academic researchers in both FPGA and Deep Learning community.

### High-Level Workflow

![High-Level-wrkfl](https://github.com/CatherineMeng/FGYM-user-demo/blob/main/fig/diag.png)
FGYM has two phases. In the pre-execution phase, the Host Code Generator takes high-level algorithm and device specifications as user inputs, and generates: (1) host executable, and (2) main memory port specifications that needs to be implemented in the RL FPGA kernel. Several template RL agents will be made available in the Pong_float_interface and Cartpole_float_interface folders. In the post-execution phase, (3) profiling results are provided to identify execution bottlenecks. Scroll down to the "Organizaion and Usage" section for detailed commands.

## Software Dependencies

[OpenAI Gym](https://gym.openai.com/docs/):
```
pip install gym
```
[PyOpenCL](https://documen.tician.de/pyopencl/misc.html#):
```
pip install pyopencl
```
[Stable-Baseline 3 (Vetcorized Gym Environments)](https://stable-baselines3.readthedocs.io/en/master/guide/install.html):
```
pip install stable-baselines3
```
Other packages to enable visualization of testing and profiling funtionalities:
```
pip install pyexcel pyexcel-xlsx
pip install numpy
pip install matplotlib
pip install tqdm
pip install prettytable
```
## Organizaion and Usage