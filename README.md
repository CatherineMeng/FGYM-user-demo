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

### Organization

The "Tookit Template" folder includes all template files for the host generator, pre-execution and post-execution programs (they are not algorithm- or benchmark-specific);

If you would like to modify an existing project as a starting point, the "Host Example" folder provides two complete examples of Deep RL algorithms on two different GYM benchmarking environments:
Cartpole(DQN) and Pong(PPO).

### (Optional) Pre-Execution

** pre-generator.py ** takes as inputs of the following:
1. the algorirthm specification file, 
2. the device specification file, 
3. the name of the target benchamrking environment;
It then calculates the memory/communication requirement of the agents-device interactions and provide a recommended interface format for the kernel. The outputs are the top-level kernel declaration format for the header file and the associating configuration file (dev.cfg). Since the output is used in collabaration of the bitstream development itself, this step is optional.

To run the pre-execution program:
```
pre-generator.py <device spec> <algo spec> <env_name> -<profile_flag (0 for disable, 1 for enable)>
```
Example:
```
cd Toolkit Template
pre-generator.py <device.in> <algo.in> PongNoFrameskip-v4 -1
```

To view the requirement for the input specification files, refer to the README doc in the Toolkit Template folder.

### Host Generator

** host_program.py ** takes as inputs of the following:
1. the algorirthm specification file, 
2. the name of the FPGA bitstream file, 
3. the name of the target benchmarking environment,
4. (optional) profiling, plotting and recording toggles
5. (optional) training/evaluation mode toggle

To run the host program (training mode):
```
host_program.py -i <algo spec> -e <env_name> -b <bitstream_name> -m <mode_flag (0 for evaluation, default: 1 for training)> -p <mode_flag (default: 0 for no plot, 1 otherwise)> -v <mode_flag (default: 0 for no video recording, 1 otherwise)>
pre-generator.py <device.in> <algo.in> PongNoFrameskip-v4 -1
```
Example:
```
cd host_examples/Pong
host_program.py -i host_params.in -b mlp_DDR_pong_1.xclbin -e PongNoFrameskip-v4 -m train -p 1
```

The evaluation mode allows evaluating the current performance os an agent without updating its policy. A recording of the agent interacting with the environment can be generated by toggling the command arguments:
```
cd host_examples/Pong
host_program.py -m eval -v 1 -i host_params.in -b mlp_DDR_pong_1.xclbin -e PongNoFrameskip-v4 
```

The host program shows a high-level profiling result at the end of execution, which returns the latency breakdown the the major Deep RL components: Policy Inferences, Environmental Steps, Training, Others.

The basic host generator provides an implementation of DQN from user-specified number of parallel rollout-agents and training batch sizes, where DNN policy is off-loaded to the FPGA. The program can then be modified for more advanced optimizations (e.g. image pre-processing for Atari - example in host_examples/Pong; training function customizations and acceleration, etc).

### Post_Execution

To view the low-level profiling results including kernel computation, PCIe and DDR performance, run the **kernel_prof.py** program.
Example: 
```
cd host_examples/Cartpole
cp ../../Toolkit\ Template/kernel_prof.py .
kernel_prof.py 
```