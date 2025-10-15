# 📁 FFT Implementations Archive

## 🎯 Структура архива

### `/Working/` - Рабочие версии
- `fft16_simple_correct_WITH_LOOP.cu` - эталон FFT16
- `fft32_simple_correct_WITH_LOOP.cu` - эталон FFT32  
- `fft64_simple_correct.cu` - FFT64
- `fft128_simple_correct.cu` - FFT128
- `fft256_simple_correct.cu` - FFT256
- `fft512_simple_correct.cu` - FFT512
- `fft1024_simple_correct.cu` - FFT1024
- `fft16_batch64.cu` - FFT16 batch
- `fft32_batch32.cu` - FFT32 batch
- `fft32_batch32_v2.cu` - FFT32 batch v2
- `fft64_batch16.cu` - FFT64 batch

### `/Experimental/` - Экспериментальные версии
- Shared Memory implementations
- Tensor Cores attempts
- Warp-level optimizations
- Matrix multiplication approaches
- Various optimization attempts

### `/CPU_Reference/` - CPU эталоны
- `fft16_cpu_reference.cpp` - CPU reference для FFT16

## 📊 Результаты
- **FFT16**: 331.2x быстрее cuFFT
- **FFT32**: 1.3x быстрее cuFFT
- **FFT64-1024**: все размеры работают корректно

---
**Архивировано**: 2025-01-10  
**Статус**: Готов к продолжению работы
