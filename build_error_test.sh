#!/bin/bash
cd /home/alex/C++/CudaCalc
g++ -std=c++17 -I. -I/usr/local/cuda-13.0/include \
    test_error_details.cpp \
    SignalGenerators/src/sine_generator.cpp \
    ModelsFunction/src/nvidia/fft/FFT16_WMMA/fft16_wmma.cpp \
    ModelsFunction/src/nvidia/fft/FFT16_WMMA/fft16_wmma_profiled.cpp \
    Tester/src/performance/basic_profiler.cpp \
    Tester/src/validation/fft_validator.cpp \
    build/lib/libModelsFunction.a \
    -L/usr/local/cuda-13.0/lib64 -lcudart -lcufft \
    -o build/error_analysis

echo "Compiled! Running..."
./build/error_analysis
