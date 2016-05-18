# CaffeMex 
v0.0-alpha0.1: 

A multiple-GPU version of MATLAB Caffe

For now it is an unstable version, which is only passed on following jobs:
```
ImageNet classification by Yu Liu
Faster-rcnn object detection by Hongwei Qin
```
Pass or Bug report: liuyu@sensetime.com
## Installation

1.Download the latest beta version of protobuf(for now it is 3.0.0 beta2).
```
    wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-cpp-3.0.0-beta-2.zip
    unzip protobuf-cpp-3.0.0-beta-2.zip
```

2.While installing protobuf：
```
    Step1: run ./configure
    Step2: edit `src/Makefile`, in `CXXFLAGS` add `-fPIC` and recompile
    Step3: enter root folder and make & make check & sudo make install
    Step4: sudo ldconfig
```

3.Make caffe with my Makefile.
```
    cd CaffeMex
    make matcaffe
```

Attention: DO NOT cover my Makefile with official's Makefile for there are some modifications in it.

## Usage Notice

0.In CaffeMex/matlab, you will find a folder `+DNN` including core mex file and a `sample.m` including some common examples.

1.You need to `set_device_id` before each time you `init_solver`.

2.You need to `release_solver` before you `clear mex` or init new solver. Otherwise MATLAB will crash due to the unsolved bug in glog-0.3.3.

3.For stabilization, DO NOT re-install glog or protobuf after make success.

4.More example will come soon.

## Interface Reference
   |Interface             |Introduction     |remark |
   | -------------------- |:---------------:|-------|
   |"forward"|	|
   |"backward"|	|
   |"init"|	|
   |"is_initialized"|	|
   |"set_mode_cpu"|	|
   |"set_mode_gpu"|	|
   |"set_device"|	|
   |"set_input_size"|	|
   |"get_response"|	|
   |"get_weights"|	|
   |"get_init_key"|	|
   |"release"|	|
   |"read_mean"|	|
   |"set_random_seed"|	|
   |"init_solver"|	|
   |"recovery_solver"|	|
   |"release_solver"|	|
   |"get_solver_iter"|	|
   |"get_solver_max_iter"|	|
   |"get_weights_solver"|	|
   |"get_response_solver"|	|
   |"set_weights_solver"|	|
   |"set_input_size_solver"|	|
   |"set_device_solver"|	|
   |"train"|	|
   |"test"|	|
   |"snapshot"|	|
   |"init_log"|	|
   |"END"|  |