# Конституция проекта CudaCalc

**Фундаментальные принципы разработки - каменные скрижали проекта**

---

## 🎯 Миссия

Создать высокопроизводительную CUDA-библиотеку для тензорных вычислений, где:
- **Производительность** - не компромисс, а требование
- **Memory safety** - гарантируется дизайном, а не дисциплиной
- **Простота использования** - не означает потерю контроля

---

## 📜 Незыблемые принципы

### 1. 🚀 Производительность превыше всего

#### 1.1 Измерять, не гадать
```
❌ "Эта оптимизация должна помочь"
✅ "Benchmark показал speedup 2.3x (50ms → 21ms)"
```

**Правила:**
- Любая оптимизация должна иметь цифры из профилировщика (nvprof, Nsight Compute)
- Baseline метрики фиксируются ДО начала оптимизации
- Performance regression тесты обязательны для критичных операций
- Документировать результаты в MemoryBank

#### 1.2 Инструменты профилирования
```bash
# Обязательно использовать ДО и ПОСЛЕ оптимизации
nsys profile --stats=true ./benchmark
ncu --metrics=all --target-processes=all ./benchmark
```

#### 1.3 Целевые метрики (RTX 3090)
- **FFT latency**: < 5ms для 100 тензоров 512x512
- **Memory overhead**: < 10% от размера данных
- **Throughput**: > 1000 тензоров/сек
- **GPU utilization**: > 85%

---

### 2. 🛡 Memory Safety - by design

#### 2.1 RAII для всего
```cpp
// ✅ Правильно - RAII гарантирует cleanup
class TensorFFT {
    struct CUFFTPlanDeleter {
        void operator()(cufftHandle* plan) {
            if (plan) cufftDestroy(*plan);
        }
    };
    std::unique_ptr<cufftHandle, CUFFTPlanDeleter> plan_;
};

// ❌ Неправильно - утечка при exception
class TensorFFT {
    cufftHandle plan_;
    ~TensorFFT() { cufftDestroy(plan_); }  // Не вызовется при exception!
};
```

**Правила:**
- Все GPU ресурсы (память, handles, streams) управляются через RAII
- Zero raw pointers в публичном API
- `cudaMalloc` → wrapped в smart pointer или RAII класс
- Exceptions безопасны (no leaks)

#### 2.2 Error checking - всегда и везде
```cpp
// ✅ Обязательно
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw CUDAException(cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while(0)

CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

// ❌ Недопустимо
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);  // Игнорируем ошибки!
```

#### 2.3 Memory leak detection
```bash
# Обязательно запускать периодически
cuda-memcheck --leak-check full ./tests
valgrind --tool=memcheck --leak-check=full ./tests
```

---

### 3. 🧪 Тестирование - не опция

#### 3.1 Unit Tests
```
Каждый публичный метод = Unit test
Каждый баг = Regression test
```

**Правила:**
- Минимум 80% code coverage для публичного API
- Google Test для всех тестов
- Тесты должны быть:
  - **Fast**: < 1 секунда на тест
  - **Isolated**: независимы друг от друга
  - **Repeatable**: детерминированные
  - **Self-validating**: pass/fail, без ручной проверки

#### 3.2 Edge Cases - обязательны
```cpp
TEST(TensorFFT, EmptyInput) { ... }         // Пустой тензор
TEST(TensorFFT, SingleElement) { ... }      // 1x1 тензор
TEST(TensorFFT, HugeTensor) { ... }         // 4096x4096 тензор
TEST(TensorFFT, NonPowerOf2) { ... }        // Должен выбросить exception
TEST(TensorFFT, NullPointer) { ... }        // nullptr handling
TEST(TensorFFT, OutOfMemory) { ... }        // Graceful failure
```

#### 3.3 Performance Tests
```cpp
// Regression тест - предотвращает замедление
BENCHMARK(BM_FFT_512x512)->Iterations(1000);
// Assert: среднее время < 1.2 * baseline
```

---

### 4. 📝 Код стиль - единообразие

#### 4.1 Форматирование
```bash
# Автоматически перед каждым коммитом
clang-format -i src/*.cu include/*.hpp
```

**Стандарт:** Google C++ Style Guide

#### 4.2 Naming Conventions
```cpp
// Classes/Structs: PascalCase
class TensorFFT { };
struct FFTConfig { };

// Functions/Methods: snake_case
void forward_fft(const Tensor& input);
size_t estimate_memory() const;

// Variables: snake_case
float* device_ptr;
size_t batch_size;

// Constants: kPascalCase
constexpr size_t kMaxBatchSize = 1024;
constexpr float kEpsilon = 1e-6f;

// Macros: UPPER_CASE
#define CUDA_CHECK(call) ...
```

#### 4.3 Документация - Doxygen
```cpp
/**
 * @brief Performs forward FFT on input tensor
 * 
 * @param input Input tensor (must be on GPU, power-of-2 dimensions)
 * @return Tensor Output tensor in frequency domain
 * 
 * @throws CUDAException If CUDA operation fails
 * @throws std::invalid_argument If input dimensions invalid
 * 
 * @note This function uses cached cuFFT plans for performance
 * @see inverse_fft() for inverse transform
 * 
 * Example:
 * @code
 * TensorFFT fft;
 * auto input = make_tensor<float>({512, 512});
 * auto output = fft.forward(input);
 * @endcode
 */
Tensor forward(const Tensor& input);
```

---

### 5. ⚡ CUDA Best Practices

#### 5.1 Синхронизация - минимум
```cpp
// ✅ Хорошо - асинхронно
cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(...);
// Продолжаем CPU работу...

// ❌ Плохо - синхронная блокировка
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);  // CPU ждет!
cudaDeviceSynchronize();  // Блокируем без необходимости!
```

#### 5.2 Memory Coalescing
```cpp
// ✅ Хорошо - coalesced access
__global__ void kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = ...;  // Соседние threads → соседние адреса
}

// ❌ Плохо - strided access
__global__ void kernel(float* data, int stride) {
    int idx = threadIdx.x * stride;  // Прыжки по памяти!
    data[idx] = ...;
}
```

#### 5.3 Occupancy - максимизировать
```bash
# Проверять occupancy для каждого kernel
ncu --metrics=sm__warps_active.avg.pct_of_peak kernel
# Цель: > 75%
```

#### 5.4 Stream Management
```cpp
// Использовать streams для параллелизма
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Параллельные операции
kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block, 0, stream2>>>(...);  // Одновременно!
```

---

### 6. 🔄 Оптимизация - итеративно

#### 6.1 Последовательность
```
1. Правильный код (корректность)
   ↓
2. Тесты (надежность)
   ↓
3. Benchmark (baseline)
   ↓
4. Профилирование (hotspots)
   ↓
5. Оптимизация (targeted)
   ↓
6. Verification (тесты still pass)
   ↓
7. Benchmark (измеряем gain)
   ↓
8. MemoryBank (документируем)
```

#### 6.2 Приоритеты оптимизации
```
1. Algorithm (O(n²) → O(n log n))
2. Memory access patterns (coalescing)
3. Occupancy (threads utilization)
4. Register usage (reduce spilling)
5. Micro-optimizations (последнее!)
```

#### 6.3 "Premature optimization is evil"
```
❌ НЕ оптимизировать:
- Код, который не в hotpath (< 5% времени)
- Код, который работает достаточно быстро
- Код, который еще не имеет тестов

✅ Оптимизировать:
- Профилировщик показал hotspot (> 20% времени)
- Не достигаем целевых метрик
- Есть очевидная неэффективность (O(n²) где можно O(n))
```

---

### 7. 🧩 API Design - простота и контроль

#### 7.1 Принцип минимального удивления
```cpp
// ✅ Интуитивно
auto result = fft.forward(input);         // Очевидно что делает
auto result = fft.inverse(frequency);     // Симметрично

// ❌ Неожиданно
auto result = fft.transform(input, true);  // true что значит?
fft.execute(input, &output, MODE_FWD);     // C-style API
```

#### 7.2 RAII и move semantics
```cpp
// ✅ Эффективно
Tensor result = fft.forward(std::move(input));  // Move, no copy

// ❌ Копирование
Tensor result = fft.forward(input);  // Копия (медленно!)
```

#### 7.3 Error handling
```cpp
// ✅ Exceptions для ошибок
try {
    auto result = fft.forward(input);
} catch (const CUDAException& e) {
    // Handle CUDA errors
} catch (const std::invalid_argument& e) {
    // Handle validation errors
}

// ❌ Error codes (C-style, не идиоматично для C++)
int err = fft_forward(input, &output);
if (err != SUCCESS) { ... }
```

---

### 8. 📚 Документация - всегда актуальная

#### 8.1 Что документировать
```
✅ Обязательно:
- Все публичные API (Doxygen)
- Алгоритмы и сложность (Big-O)
- Предусловия и постусловия
- Thread safety гарантии
- Performance характеристики

✅ Желательно:
- Примеры использования
- Edge cases
- Known limitations

❌ Не нужно:
- Очевидные вещи (геттеры/сеттеры)
- Implementation details приватных методов
```

#### 8.2 Где документировать
```
- CLAUDE.md - общий контекст проекта
- constitution.md - принципы (этот файл)
- spec.md - спецификация фичи
- plan.md - детали реализации
- Code comments - Doxygen для публичного API
- MemoryBank - инсайты и lessons learned
```

---

### 9. 🔐 Безопасность и корректность

#### 9.1 Input validation
```cpp
Tensor forward(const Tensor& input) {
    // Валидация ВСЕГДА первая
    if (input.empty()) {
        throw std::invalid_argument("Input tensor is empty");
    }
    if (!is_power_of_2(input.width()) || !is_power_of_2(input.height())) {
        throw std::invalid_argument("Dimensions must be power of 2");
    }
    if (!input.is_on_device()) {
        throw std::invalid_argument("Input must be on GPU");
    }
    
    // Теперь можно работать
    ...
}
```

#### 9.2 Assertions для invariants
```cpp
void TensorFFT::execute() {
    assert(plan_ != nullptr && "FFT plan must be initialized");
    assert(input_size_ > 0 && "Input size must be positive");
    CUDA_CHECK(cufftExecC2C(plan_, ...));
}
```

#### 9.3 Thread safety
```cpp
// Документировать thread-safety явно
/**
 * @brief Thread-safe FFT executor
 * @threadsafety This class is thread-safe for concurrent read operations.
 *               Writes must be externally synchronized.
 */
class TensorFFT {
    mutable std::mutex mutex_;
    // ...
};
```

---

### 10. 🔄 Git workflow

#### 10.1 Коммиты
```bash
# Маленькие, атомарные коммиты
git commit -m "feat(fft): add forward FFT for complex tensors"
git commit -m "test(fft): add unit tests for forward FFT"
git commit -m "perf(fft): cache cuFFT plans for 3x speedup"

# ❌ НЕ делать
git commit -m "fixed stuff"
git commit -m "WIP"  # (только в личных ветках)
```

#### 10.2 Ветвление
```
main (production-ready)
  ↓
develop (integration)
  ↓
feature/fft-optimization (фича)
```

#### 10.3 Code Review
```
Обязательно ревью для:
- Любые изменения публичного API
- Performance критичный код
- Безопасность и memory management
```

---

## 🎓 Обучение и рост

### Ресурсы для команды
- **CUDA Programming**: NVIDIA Developer Blog
- **C++ Best Practices**: https://github.com/cpp-best-practices
- **Performance**: "CUDA C++ Best Practices Guide"

### Регулярные активности
- **Code Review** - каждый PR
- **Профилирование** - раз в спринт
- **Refactoring** - когда накопились tech debt
- **Learning** - изучать новые CUDA features

---

## ⚖️ Исключения из правил

**Правила можно нарушить ТОЛЬКО если:**
1. Есть документированная причина (performance, legacy compatibility)
2. Задокументировано в коде (`// NOLINTNEXTLINE: reason`)
3. Обсуждено с командой (или в MemoryBank для solo проектов)

**Пример:**
```cpp
// NOLINTNEXTLINE: raw pointer required for cuFFT C API
cufftReal* raw_ptr = tensor.data();
CUDA_CHECK(cufftExecR2C(plan_, raw_ptr, output));
```

---

## 📊 Метрики качества

### Code Quality
- Code coverage: >= 80%
- Static analysis warnings: 0
- Memory leaks: 0
- Clang-tidy issues: 0

### Performance
- GPU utilization: > 85%
- Memory overhead: < 10%
- FFT latency: < target
- Throughput: > target

---

## 🔄 Эволюция конституции

**Этот документ живой**, но изменяется медленно:
- Добавление принципов: требует обоснования
- Изменение принципов: только если принцип не работает
- Удаление: крайне редко

**История изменений:**
- 2025-10-09: Создание v1.0 (AlexLan73)

---

**Этот документ - фундамент проекта. AI и разработчики должны следовать этим принципам автоматически.**

**Версия:** 1.0  
**Проект:** CudaCalc  
**Автор:** AlexLan73  
**Статус:** ACTIVE (незыблемые принципы)

