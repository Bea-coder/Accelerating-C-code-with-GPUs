# Accelerating-C-code-with-GPUs

L6118 - Simple Steps to Speed Up Your C Code Dramatically with GPU


Many scientists have developed their own old trusted C code, and they would like to make it run faster using the new GPU technology. However, they don't how to start this process. In this tutorial, we provide simple steps to speed up a piece of C code. To do so, we have selected the most common algorithms where the parallel architecture benefits are best and we describe the little details encountered in the transition from C to CUDA programming.
This tutorial will cover how to do:
* Allocate Dynamic memory in the device 
* Call functions 
* Data transfer from device to host and host to device
* Execution of kernels
* Implementation of averages and histograms

We organize the tutorial in the following manner:
* Review of CUDA architecture and parallel programming.
* Reduce and Scan Algorithms
* Examples and Exercises:
