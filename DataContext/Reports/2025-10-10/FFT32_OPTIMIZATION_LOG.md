# FFT32 Optimization Experiments Log
## Date: 2025-10-10
## Goal: Achieve 0.0488ms or better for FFT32×16384 windows

---

## BASELINE (Current Implementation)

**Configuration:**
- Block dimension: [32, 32] = 1024 threads
- FFT per block: 32
- Total blocks: 512 (16384/32)
- Shared memory: [32][34] with padding

**Results:**
- Min compute: 0.10605 ms
- Median: 0.39715 ms
- Mean: 0.41522 ms
- **Status: SLOWER by 117% ❌**

**Analysis:**
- Good for small data (256 windows): 0.01024ms ✅
- Poor scaling to large data
- Possible issues: low GPU occupancy, memory bandwidth

---

## EXPERIMENT 1: V2 Ultra - 1D Indexing + Dynamic Shared Memory

**Hypothesis:** Old project style (1D + dynamic shared memory) will be faster

**Configuration:**
- 1D thread indexing: threadId = fftId * 32 + pointId
- 1024 threads per block (32 FFT × 32 points)
- Dynamic shared memory
- Only half threads work (if pointId < 16)

**Results:**
- Min compute: **10.91603 ms** ❌❌❌
- **100x SLOWER than V1!**
- Validation: FAILED

**Analysis:**
- Dynamic shared memory is MUCH slower!
- 1D indexing inefficient
- Half threads idle = wasted GPU resources
- **CONCLUSION: DO NOT USE THIS APPROACH!**

---

## EXPERIMENT 2: [128, 8] Configuration

**Hypothesis:** Maximum FFT per block for best occupancy

**Configuration:**
- Block dimension: [128, 8] = 1024 threads
- FFT per block: 128
- Total blocks: 128 (16384/128)

**Status:** Pending...

---

## EXPERIMENT 3: Optimized Padding

**Hypothesis:** Better padding reduces bank conflicts

**Status:** Pending...

---

## NEXT EXPERIMENTS

### EXPERIMENT 3: Optimize V1 (2D blocks)
- Keep [32, 32] configuration (it's fastest so far!)
- Try optimizations:
  - Better memory access patterns
  - Reduce __syncthreads() calls
  - Optimize twiddle application
  - Check if accuracy fixes help speed

### EXPERIMENT 4: Check OLD project implementation details
- Maybe they use different algorithm?
- Check their butterfly pattern
- Check their twiddle application

---

## BEST RESULT SO FAR

**V1 (2D [32,32] configuration)**: 0.10605 ms
- Still 2x slower than target (0.0488ms)
- But 100x FASTER than V2!

**Next step:** Optimize V1 instead of trying completely different approaches

---

