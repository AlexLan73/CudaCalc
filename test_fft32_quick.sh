#!/bin/bash
# Quick test for different FFT32 versions

echo "=== TESTING V1 (Baseline [32,32]) ==="
./build/bin/cudacalc_fft32_test 2>&1 | grep -A 5 "Min:"

echo ""
echo "=== TESTING V3 (Like FFT16 [64,16]) ==="
./build/bin/cudacalc_fft32_multisize 2>&1 | grep "16384"

