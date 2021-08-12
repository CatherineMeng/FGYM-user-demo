# General Resources
General instructions, formatting of the input spec files, etc

## Input 1: Algorithmic Parameters

The algorithmic input file <algo.in> should specify common hyper-parameters shared by most deep reinforcement algorithms. Their meanings are explained as the following:
```
N: This is the number of parallel agents deployed, which is equivalent to the policy inference batch size in a fully synchronous setting supported by subVecEnv
T: This is the rollout trajectory length, that is, how many environment steps are executed by each one of the parallel agents in one iteration of RL
M: This is the training batch size
K: This is the number of training epoch on the same batch of experiences each time the training function is called
```
They should be put in the exact order specified in the example **algo.in** file, each followed by a space and an integer value.
Note that N and M specifically impact the kernel interface. Therefore, N and M are required for the pre-execution program, and all are required for the **host_.py** program.

## Input 2: Device Specifications

The device input file <device.in> should specify device name and its associated main memory options. 
The first line 
```
platform=xilinx_u200_xdma_201830_2
```
is the first line in your kernel .cfg file. The device name can be obtained at the device official website under "Getting Started -> Vitis design flow" tab and downloading the Development Target Platform package.

Then, the name and number of banks for all available external DDR memories on your board. For example, 4 DDR banks and 2 HBM banks:
```
DDR 4 
HBM 2
```

Some platforms support [PLRAM](https://japan.xilinx.com/html_docs/xilinx2020_1/vitis-guidance/PLRAM_USAGE.html), which uses part of the on-chip SRAM as the global memory when communicating data through PCIe. This gives lower-latency data migration than using DDR as the global memory. The pre-execution program will help decide whether the observation data of a certain gym environ,ent can take advantage of such functionalitis. To enable this, list the size of availanle PLRAM storag ein KB (put 0 if no PLRAM support):
```
PLRAM 128
```
