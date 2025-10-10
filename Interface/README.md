# Interface Module

**Type:** Header-only library  
**Dependencies:** None  
**Purpose:** Base interfaces and data structures for all CudaCalc modules

---

## Files

### signal_data.h
Defines signal data structures:
- `StrobeConfig` - strobe configuration
- `InputSignalData` - input signal (HOST memory)
- `OutputSpectralData` - output spectrum (HOST memory)

### igpu_processor.h
Defines base interface:
- `IGPUProcessor` - abstract class for all GPU implementations

### common_types.h
Common utilities:
- `CUDA_CHECK()` macro for error checking
- `CUFFT_CHECK()` macro for cuFFT errors
- Constants (kPI, k2PI, etc.)
- GPU info functions

---

## Usage

```cpp
#include "Interface/include/signal_data.h"
#include "Interface/include/igpu_processor.h"

// Use structures
StrobeConfig config{4, 1024, 16};
InputSignalData input;
input.config = config;
// ...
```

---

**Version:** 1.0  
**Status:** ✅ Complete (TASK-002)

