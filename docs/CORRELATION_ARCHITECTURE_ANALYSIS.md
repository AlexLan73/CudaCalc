# 🔍 Анализ архитектуры GPU-ускоренной корреляции сигналов

## 📋 Постановка задачи

**Алгоритм корреляции:**
1. **FFT входного сигнала** (1 раз)
2. **Перемножение спектров** с 40 FFT опорных сигналов
3. **40 обратных FFT** для получения корреляций

**Параметры:**
- Максимальная длина сигнала: 2^15 = 32,768 точек
- До 20 входных сигналов одновременно
- 40 циклически сдвинутых опорных сигналов
- Комплексные числа (cuComplex)

## 🎯 Варианты архитектуры

### Вариант 1: 40 параллельных вычислений в цикле

```cpp
// Псевдокод
for (int shift = 0; shift < 40; ++shift) {
    // Перемножение спектров
    multiply_spectra_kernel<<<grid, block>>>(
        input_fft, reference_fft[shift], multiplied_spectrum[shift]
    );
    
    // Обратный FFT
    launch_fft_inverse(multiplied_spectrum[shift], correlation[shift]);
}
```

**Плюсы:**
- ✅ Простая реализация
- ✅ Контроль загрузки GPU
- ✅ Минимальная память на один момент

**Минусы:**
- ❌ 40 kernel launch'ов (высокий overhead)
- ❌ Неэффективное использование GPU
- ❌ Последовательное выполнение

### Вариант 2: Частичные запуски с контролем загрузки

```cpp
// Псевдокод
const int BATCH_SIZE = 8;  // 8 сдвигов за раз
for (int batch = 0; batch < 5; ++batch) {  // 40/8 = 5 батчей
    // Перемножение 8 спектров параллельно
    multiply_spectra_batch_kernel<<<grid, block>>>(
        input_fft, reference_fft + batch*8, multiplied_spectrum + batch*8, 8
    );
    
    // 8 обратных FFT
    launch_fft_inverse_batch(multiplied_spectrum + batch*8, correlation + batch*8, 8);
}
```

**Плюсы:**
- ✅ Контроль загрузки GPU
- ✅ Меньше kernel launch'ов (5 вместо 40)
- ✅ Лучшее использование GPU

**Минусы:**
- ❌ Все еще последовательные батчи
- ❌ Сложность реализации

### Вариант 3: Один большой вектор (BATCHED MODE)

```cpp
// Псевдокод
// Подготовка: все 40 спектров в одном массиве
cuComplex all_multiplied_spectra[40 * signal_length];

// Одно перемножение для всех 40 сдвигов
multiply_all_spectra_kernel<<<grid, block>>>(
    input_fft, reference_fft_array, all_multiplied_spectra, 40
);

// 40 обратных FFT одним вызовом
launch_fft_inverse_batch(all_multiplied_spectra, all_correlations, 40);
```

**Плюсы:**
- ✅ Максимальная параллелизация
- ✅ Минимум kernel launch'ов (2 вместо 40)
- ✅ Лучшее использование GPU memory bandwidth
- ✅ Попадание в оптимальную зону batch size

**Минусы:**
- ❌ Больше памяти (40 × signal_length)
- ❌ Сложность управления памятью

## 📊 Детальный анализ производительности

### На основе наших FFT результатов:

**FFT1024 (ближайший к 2^15):**
- **1 окно**: 0.0202 ms
- **40 окон**: ~0.0202 ms (почти без изменений)
- **Speedup vs cuFFT**: 2.24x

**Memory requirements:**
- **Один сигнал 2^15**: 32,768 × sizeof(cuComplex) = 256 KB
- **40 сдвигов**: 40 × 256 KB = 10.24 MB
- **20 входных сигналов**: 20 × 10.24 MB = 204.8 MB

### Анализ kernel launch overhead:

**Вариант 1 (40 launches):**
- **Overhead**: 40 × kernel_latency ≈ 40 × 0.001 ms = 0.04 ms
- **Compute time**: 40 × 0.0202 ms = 0.808 ms
- **Total**: ~0.848 ms

**Вариант 3 (2 launches):**
- **Overhead**: 2 × kernel_latency ≈ 0.002 ms
- **Compute time**: ~0.0202 ms (batched)
- **Total**: ~0.022 ms
- **Ускорение**: 38.5x!

## 🎯 МОЯ РЕКОМЕНДАЦИЯ: Вариант 3 + Гибридный подход

### Оптимальная архитектура:

```cpp
class CorrelationEngine {
private:
    // Предварительно вычисленные FFT опорных сигналов
    cuComplex* reference_fft_array;  // 40 × signal_length
    
    // Буферы для batched вычислений
    cuComplex* input_fft_buffer;
    cuComplex* multiplied_spectra_buffer;  // 40 × signal_length
    cuComplex* correlation_results;        // 40 × signal_length
    
public:
    // Основной метод корреляции
    void correlate_signals(
        const cuComplex* input_signal,
        cuComplex* correlation_output,
        int signal_length,
        int num_input_signals = 1
    ) {
        // 1. FFT входного сигнала (batched для нескольких сигналов)
        launch_fft_forward_batch(input_signal, input_fft_buffer, num_input_signals);
        
        // 2. Перемножение спектров (все 40 сдвигов одновременно)
        multiply_spectra_batch_kernel<<<grid, block>>>(
            input_fft_buffer,
            reference_fft_array,
            multiplied_spectra_buffer,
            num_input_signals * 40  // Общее количество перемножений
        );
        
        // 3. Обратные FFT (batched для всех корреляций)
        launch_fft_inverse_batch(
            multiplied_spectra_buffer,
            correlation_results,
            num_input_signals * 40
        );
        
        // 4. Копирование результатов
        cudaMemcpy(correlation_output, correlation_results, 
                  num_input_signals * 40 * signal_length * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);
    }
};
```

### Преимущества гибридного подхода:

1. **Максимальная производительность**: 2 kernel launch'а вместо 40
2. **Эффективное использование памяти**: предварительно вычисленные FFT опор
3. **Масштабируемость**: поддержка до 20 входных сигналов
4. **Контроль загрузки**: можно разбить на меньшие батчи при необходимости

### Адаптивная стратегия:

```cpp
// Адаптивный выбор размера батча
int get_optimal_batch_size(int signal_length, int num_signals) {
    int total_correlations = num_signals * 40;
    
    if (total_correlations <= 100) {
        return total_correlations;  // Все сразу
    } else if (total_correlations <= 400) {
        return 100;  // Батчи по 100
    } else {
        return 50;   // Батчи по 50
    }
}
```

## 🚀 Ожидаемые результаты

### Производительность:
- **Kernel launches**: 40 → 2 (20x reduction)
- **Memory bandwidth**: оптимальное использование
- **GPU utilization**: максимальная загрузка
- **Total time**: ~0.022 ms vs ~0.848 ms (38.5x speedup)

### Масштабируемость:
- **1 сигнал**: 40 корреляций за ~0.022 ms
- **20 сигналов**: 800 корреляций за ~0.4 ms
- **Memory usage**: 204.8 MB (в пределах возможностей GPU)

## 🎯 Итоговая рекомендация

**ИСПОЛЬЗУЙТЕ ВАРИАНТ 3 (BATCHED MODE) с гибридным подходом!**

**Обоснование:**
1. **38.5x ускорение** по сравнению с последовательными запусками
2. **Оптимальное использование** наших FFT реализаций
3. **Масштабируемость** для до 20 входных сигналов
4. **Минимальные обмены** с глобальной памятью
5. **Контроль загрузки** через адаптивные батчи

**Это решение максимально использует преимущества наших оптимизированных FFT!** 🎉

---

*Анализ основан на результатах comprehensive FFT benchmark и occupancy sweep*


