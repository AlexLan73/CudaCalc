# Спецификация: FFT16 Baseline Testing Pipeline

**Автор:** AlexLan73  
**Дата:** 09 октября 2025  
**Статус:** Draft → Review  
**Приоритет:** Высокий (первая реализация)

---

## 1. Обзор

### 1.1 Проблема
Необходимо создать **базовую тестовую цепочку** для проверки производительности FFT на 16 точек - фундаментальный примитив для обработки сигналов в CudaCalc.

**Текущая ситуация:**
- Есть существующий код с FFT kernels (TensorFFTKernels.cu)
- Нет унифицированной системы тестирования
- Нет автоматического профилирования
- Нет валидации результатов
- Нет системы логирования

**Проблемы:**
- Невозможно объективно сравнить разные реализации FFT
- Нет данных о производительности на целевом железе (RTX 3060)
- Нет гарантии корректности результатов

### 1.2 Решение
Создать **полную тестовую цепочку** для FFT16:
1. **Генератор тестовых сигналов** (синусоиды с заданными параметрами)
2. **Две реализации FFT16** для сравнения скорости:
   - Вариант A: Tensor Cores (wmma) с FP16
   - Вариант B: Обычный 2D shared memory с FP32
3. **Профилирование** через CUDA Events (upload, compute, download)
4. **Валидация** через сравнение с cuFFT
5. **JSON логирование** всех результатов

### 1.3 Цели
- **Цель 1**: Создать базовую архитектуру для всех будущих тестов
- **Цель 2**: Определить оптимальный подход для FFT16 (Tensor Cores vs обычный)
- **Цель 3**: Получить baseline метрики производительности на RTX 3060
- **Цель 4**: Верифицировать корректность обеих реализаций
- **Цель 5**: Настроить CMake для сборки на Ubuntu

---

## 2. Требования

### 2.1 Функциональные требования

**FR-1: Генерация тестовых сигналов**
- Тип сигнала: синусоида (комплексная)
- Параметры строба:
  - Количество лучей: 4
  - Точек на луч: 1024
  - Всего точек: 4096 комплексных чисел
- Параметры синуса:
  - Период: 8 точек (половина окна FFT16)
  - Амплитуда: 1.0
  - Начальная фаза: 0.0
- Формула: `signal[n] = exp(i * 2π * n / 8)`
- Приоритет: **Высокий**

**FR-2: Две реализации FFT16**
- **Вариант A - Tensor Cores (wmma):**
  - Использование FP16 (half precision)
  - Tensor Cores для ускорения butterfly операций
  - Линейная раскрутка 4 stages (без цикла)
  - Организация: 64 FFT в одном блоке
  - fftshift применяется в kernel при записи
- **Вариант B - Обычный 2D:**
  - FP32 точность
  - Shared memory как 2D массив `[64 FFTs][16 points]`
  - Линейная раскрутка 4 stages
  - 64 FFT в блоке
  - fftshift в kernel
- Общее:
  - Размер FFT: 16 точек
  - Количество окон: 256 (4096 / 16)
  - Количество блоков: 4 (256 FFT / 64)
- Приоритет: **Критический**

**FR-3: Профилирование производительности (ДВА вида)**

**Вариант 1 - Базовое профилирование (CUDA Events):**
- Три замера через `cudaEvent_t`:
  1. **Upload time**: Host → Device (cudaMemcpy)
  2. **Compute time**: Kernel execution
  3. **Download time**: Device → Host (cudaMemcpy)
- Дополнительно логируем:
  - Дата и время теста
  - GPU модель: "NVIDIA RTX 3060"
  - CUDA версия: "13.0"
  - Драйвер версия
  - Параметры теста (лучи, точки, wFFT)
- **Статус:** Реализуем в первую очередь (baseline)

**Вариант 2 - Расширенное профилирование (Memory + Performance):**
- GPU utilization (%)
- Memory usage:
  - Allocated VRAM (MB)
  - Peak VRAM usage (MB)
  - Memory bandwidth utilization (GB/s)
- Occupancy (%)
- Warp efficiency (%)
- **Статус:** Добавим после baseline (опционально)

**Итого:** Два профайлера - BasicProfiler (CUDA Events) и MemoryProfiler (VRAM, bandwidth)
- Приоритет: **Высокий** (Вариант 1), **Средний** (Вариант 2)

**FR-4: Валидация результатов**
- Reference реализация: cuFFT (batch FFT)
- Управление: флаг `is_validate` (bool)
- Если `is_validate == true`:
  - Вычисляем FFT через cuFFT для всего сигнала
  - Сравниваем с результатом нашего kernel
  - Метрика: относительная ошибка < 0.01%
  - Логируем max error, mean error, passed/failed
- Если `is_validate == false`:
  - Валидация пропускается, данные NULL
- Приоритет: **Высокий**

**FR-5: JSON логирование**
- Все результаты сохраняются в JSON
- Формат (см. раздел 4.4)
- Путь: `DataContext/Reports/fft16_test_YYYY_MM_DD_HH_MM_SS.json`
- Приоритет: **Средний**

**FR-6: 🔴 ОБЯЗАТЕЛЬНО! Система архивирования моделей**
- **КРИТИЧНО:** Результаты экспериментов НЕ ДОЛЖНЫ затираться!
- При каждом эксперименте сохраняем:
  - Исходный код (.cu, .cpp файлы)
  - Результаты профилирования (results.json)
  - Результаты валидации (validation.json)
  - Описание эксперимента (description.txt)
- Структура: `DataContext/Models/NVIDIA/FFT/16/model_YYYY_MM_DD_vN/`
- Версионирование: автоматическое инкрементирование v1, v2, v3, ...
- Функционал `ModelArchiver`:
  - `save_model()` - сохранить текущую модель
  - `load_model()` - загрузить старую модель
  - `compare_models()` - сравнить несколько моделей
  - `list_models()` - список всех моделей
- Приоритет: **🔴 КРИТИЧЕСКИЙ - ОБЯЗАТЕЛЬНО!**

### 2.2 Нефункциональные требования

**NFR-1: Производительность**
- **Главное требование:** МАКСИМАЛЬНАЯ СКОРОСТЬ выполнения FFT16
- Compute time < 1.0 ms для 256 FFT16 на RTX 3060 (target)
- Upload/Download time < 0.5 ms каждый
- Total latency < 2.0 ms для полного цикла

**NFR-2: Точность**
- Валидация: относительная ошибка < 0.01% vs cuFFT
- FP16 вариант: допустимая потеря точности из-за half precision
- FP32 вариант: максимальная точность

**NFR-3: Модульность**
- Генератор сигналов: отдельный модуль `SignalGenerators/`
- Легкое добавление новых типов сигналов (quadrature, modulated, noise)
- Интерфейсы определены в `Interface/`

**NFR-4: Совместимость**
- Платформа: Ubuntu Linux (primary)
- CUDA: 13.x
- GPU: RTX 3060 (Compute Capability 8.6)
- CMake: 3.20+
- C++ Standard: C++17/C++20

**NFR-5: Расширяемость**
- Архитектура позволяет легко добавить:
  - Новые размеры FFT (32, 64, 128, ...)
  - Новые типы сигналов
  - Новые метрики профилирования
  - Поддержку AMD GPU (будущее)

---

## 3. Терминология

### Строб (Strobe)
**Строб** - базовая единица данных, состоящая из `k` лучей длиной `n` элементов каждый.

Для этого теста:
- k = 4 луча
- n = 1024 точки на луч
- Всего: 4096 комплексных точек

### Луч (Ray/Beam)
**Луч** - отдельный независимый набор комплексных точек (аналог канала в многоканальном осциллографе).

### Окно FFT (FFT Window)
**Окно FFT (wFFT)** - размер одного FFT преобразования.

Для этого теста: wFFT = 16 точек

### Обработка
Строб (4096 точек) разбивается на окна:
- Количество окон: 4096 / 16 = **256 окон FFT16**
- Обработка: последовательно, весь сигнал как один поток

---

## 4. Архитектура и дизайн

### 4.1 Модульная структура

```
CudaCalc/
├── Interface/                      # Интерфейсные определения
│   ├── include/
│   │   ├── igpu_processor.h       # Базовый интерфейс GPU обработки
│   │   ├── signal_data.h          # Структуры данных сигналов
│   │   └── spectral_data.h        # Структуры спектральных данных
│   └── CMakeLists.txt
│
├── SignalGenerators/               # НОВЫЙ МОДУЛЬ - генераторы сигналов
│   ├── include/
│   │   ├── signal_types.h         # enum SignalType
│   │   ├── base_generator.h       # Базовый интерфейс генератора
│   │   ├── sine_generator.h       # Генератор синусоид
│   │   └── strobe_config.h        # Конфигурация строба
│   ├── src/
│   │   ├── base_generator.cpp
│   │   └── sine_generator.cpp
│   └── CMakeLists.txt
│
├── DataContext/                    # Управление данными
│   ├── include/
│   │   ├── data_manager.h
│   │   ├── json_logger.h          # JSON логирование
│   │   └── model_archiver.h       # 🔴 ОБЯЗАТЕЛЬНО: архивирование моделей
│   ├── src/
│   │   ├── data_manager.cpp
│   │   ├── json_logger.cpp
│   │   └── model_archiver.cpp     # 🔴 ОБЯЗАТЕЛЬНО
│   ├── Reports/                    # JSON отчёты
│   ├── Models/                     # 🔴 ОБЯЗАТЕЛЬНО: Архив моделей и результатов
│   │   └── NVIDIA/
│   │       └── FFT/
│   │           └── 16/
│   │               ├── model_2025_10_09_v1/
│   │               │   ├── fft16_wmma.cu       # Исходный код
│   │               │   ├── fft16_wmma.cpp
│   │               │   ├── description.txt     # Описание эксперимента
│   │               │   ├── results.json        # Результаты профилирования
│   │               │   └── validation.json     # Результаты валидации
│   │               └── model_2025_10_09_v2/    # Следующая версия
│   └── CMakeLists.txt
│
├── ModelsFunction/                 # Экспериментальные модели
│   ├── include/
│   │   └── nvidia/
│   │       └── fft/
│   │           ├── fft16_wmma.h       # Tensor Cores версия
│   │           └── fft16_shared2d.h   # 2D shared memory версия
│   ├── src/
│   │   └── nvidia/
│   │       └── fft/
│   │           ├── fft16_wmma.cu      # Kernel wmma
│   │           ├── fft16_wmma.cpp     # Wrapper
│   │           ├── fft16_shared2d.cu  # Kernel 2D
│   │           └── fft16_shared2d.cpp # Wrapper
│   └── CMakeLists.txt
│
├── Tester/                         # Система тестирования
│   ├── include/
│   │   ├── performance/
│   │   │   ├── basic_profiler.h       # Базовое профилирование (CUDA Events)
│   │   │   ├── memory_profiler.h      # Расширенное (Memory + GPU)
│   │   │   └── profiling_data.h       # Структуры данных профайлинга
│   │   └── validation/
│   │       ├── base_validator.h       # Базовый класс валидатора
│   │       └── fft_validator.h        # Валидатор FFT через cuFFT
│   ├── src/
│   │   ├── performance/
│   │   │   ├── basic_profiler.cpp     # Обязательно (baseline)
│   │   │   └── memory_profiler.cpp    # Опционально (расширенный)
│   │   └── validation/
│   │       ├── base_validator.cpp
│   │       └── fft_validator.cpp
│   └── CMakeLists.txt
│
├── MainProgram/                    # Главное приложение
│   ├── src/
│   │   └── main_fft16_test.cpp    # Точка входа для FFT16 теста
│   └── CMakeLists.txt
│
└── CMakeLists.txt                  # Корневой CMake
```

### 4.2 Интерфейсы (Interface/)

#### signal_data.h
```cpp
#pragma once
#include <complex>
#include <vector>

namespace CudaCalc {

// Конфигурация строба
struct StrobeConfig {
    int ray_count;          // Количество лучей (4)
    int points_per_ray;     // Точек на луч (1024)
    int window_fft;         // Размер окна FFT (16)
    
    int total_points() const {
        return ray_count * points_per_ray;
    }
    
    int num_windows() const {
        return total_points() / window_fft;
    }
};

// Входной сигнал (HOST memory, CPU)
struct InputSignalData {
    std::vector<std::complex<float>> signal;  // Весь сигнал: 4096 точек
    StrobeConfig config;                      // Конфигурация
    
    // Примечание: Device memory управляется внутри реализации FFT,
    // не выставляется в публичный API
};

// Выходные спектральные данные (чистый интерфейс)
struct OutputSpectralData {
    // output[окно][спектр]: 256 окон × 16 спектров
    std::vector<std::vector<std::complex<float>>> windows;
    
    // Примечание: 
    // - StrobeConfig не нужен (мы знаем что получаем)
    // - Device memory не нужен (внутренняя реализация)
};

// Данные валидации (ГЕНЕРИРУЕТСЯ В DataContext через cuFFT)
// Если is_validate = false, то это nullptr
struct ValidationData {
    bool enabled;                             // Включена ли валидация
    
    // Эталонный результат от cuFFT:
    // reference[окно][спектр]: 256 окон × 16 гармоник
    std::vector<std::vector<std::complex<float>>> reference;
    
    // Результаты сравнения (заполняется в Tester)
    double max_error;       // Максимальная ошибка
    double mean_error;      // Средняя ошибка
    double tolerance;       // Допустимая ошибка (0.01%)
    bool passed;            // Тест пройден?
};

// Полный пакет данных для теста
struct TestDataPackage {
    InputSignalData input;
    ValidationData validation;  // Может быть disabled
};

} // namespace CudaCalc
```

#### igpu_processor.h
```cpp
#pragma once
#include "signal_data.h"

namespace CudaCalc {

class IGPUProcessor {
public:
    virtual ~IGPUProcessor() = default;
    
    virtual bool initialize() = 0;
    virtual void cleanup() = 0;
    
    virtual OutputSpectralData process(const InputSignalData& input) = 0;
    
    virtual std::string get_name() const = 0;  // "FFT16_WMMA" или "FFT16_Shared2D"
};

} // namespace CudaCalc
```

### 4.3 Генератор сигналов (SignalGenerators/)

#### signal_types.h
```cpp
#pragma once

namespace CudaCalc {

enum class SignalType {
    SINE,           // Синусоида (текущая реализация)
    QUADRATURE,     // Квадратурный сигнал (будущее)
    MODULATED,      // Модулированный (будущее)
    PULSE_MOD,      // Импульсно-модулированный (будущее)
    GAUSSIAN_NOISE, // Гауссовский шум (будущее)
    CUSTOM          // Пользовательский
};

} // namespace CudaCalc
```

#### sine_generator.h
```cpp
#pragma once
#include "Interface/signal_data.h"
#include "signal_types.h"

namespace CudaCalc {

class SineGenerator {
private:
    int ray_count_;
    int points_per_ray_;
    int period_;              // Период синуса в точках
    float amplitude_;         // Амплитуда (по умолчанию 1.0)
    float phase_;             // Начальная фаза (по умолчанию 0.0)
    
public:
    SineGenerator(int ray_count, int points_per_ray, int period,
                  float amplitude = 1.0f, float phase = 0.0f);
    
    // Генерация сигнала
    InputSignalData generate(int window_fft);
    
    // Генерация с валидацией
    TestDataPackage generate_with_validation(int window_fft, bool enable_validation);
    
    SignalType get_type() const { return SignalType::SINE; }
};

} // namespace CudaCalc
```

**Реализация:**
```cpp
InputSignalData SineGenerator::generate(int window_fft) {
    InputSignalData data;
    data.config.ray_count = ray_count_;
    data.config.points_per_ray = points_per_ray_;
    data.config.window_fft = window_fft;
    
    int total = data.config.total_points();
    data.signal.resize(total);
    
    // Генерация синуса для всего строба
    for (int n = 0; n < total; ++n) {
        float angle = 2.0f * M_PI * n / period_ + phase_;
        data.signal[n] = std::complex<float>(
            amplitude_ * std::cos(angle),
            amplitude_ * std::sin(angle)
        );
    }
    
    return data;
}
```

### 4.4 🔴 ОБЯЗАТЕЛЬНО! Система архивирования моделей

#### model_archiver.h
```cpp
#pragma once
#include <string>
#include <vector>
#include <filesystem>

namespace CudaCalc {

struct ModelInfo {
    std::string gpu_type;       // "NVIDIA"
    std::string algorithm;      // "FFT"
    int size;                   // 16
    std::string version;        // "model_2025_10_09_v1"
    std::string description;    // Описание эксперимента
    
    std::filesystem::path get_path() const;
};

// 🔴 КРИТИЧЕСКИЙ КЛАСС - предотвращает потерю результатов!
class ModelArchiver {
private:
    std::filesystem::path base_path_;  // DataContext/Models/
    
public:
    ModelArchiver(const std::string& base_path = "DataContext/Models");
    
    // Сохранить модель (исходники + результаты)
    bool save_model(const ModelInfo& info,
                   const std::vector<std::string>& source_files,
                   const std::string& results_json,
                   const std::string& validation_json);
    
    // Загрузить старую модель
    ModelInfo load_model(const std::string& version);
    
    // Сравнить несколько моделей
    std::string compare_models(const std::vector<std::string>& versions);
    
    // Список всех моделей
    std::vector<ModelInfo> list_models(const std::string& gpu_type,
                                      const std::string& algorithm,
                                      int size);
    
    // Автоинкремент версии
    std::string get_next_version(const std::string& gpu_type,
                                 const std::string& algorithm,
                                 int size);
};

} // namespace CudaCalc
```

**Использование:**
```cpp
ModelArchiver archiver;

// После успешного теста
ModelInfo info;
info.gpu_type = "NVIDIA";
info.algorithm = "FFT";
info.size = 16;
info.version = archiver.get_next_version("NVIDIA", "FFT", 16);  // v1, v2, ...
info.description = "Эксперимент с оптимизацией twiddle factors";

archiver.save_model(info,
    {"fft16_wmma.cu", "fft16_wmma.cpp"},  // Исходники
    results_json,                          // Профилирование
    validation_json                        // Валидация
);
```

### 4.5 Профилирование (Tester/) - ДВА ПРОФАЙЛЕРА

#### Вариант 1: BasicProfiler (CUDA Events) - Базовый

**basic_profiler.h**
```cpp
#pragma once
#include <cuda_runtime.h>
#include <string>

namespace CudaCalc {

// Базовое профилирование через CUDA Events
struct BasicProfilingResult {
    float upload_ms;      // Host → Device
    float compute_ms;     // Kernel execution
    float download_ms;    // Device → Host
    float total_ms;       // Total
    
    // Metadata
    std::string gpu_name;      // "NVIDIA RTX 3060"
    std::string cuda_version;  // "13.0"
    std::string driver_version;
    std::string timestamp;     // "2025-10-09T10:30:45"
    std::string algorithm;     // "FFT16_WMMA" или "FFT16_Shared2D"
    
    StrobeConfig config;
};

class BasicProfiler {
private:
    cudaEvent_t start_upload_, end_upload_;
    cudaEvent_t start_compute_, end_compute_;
    cudaEvent_t start_download_, end_download_;
    
public:
    BasicProfiler();
    ~BasicProfiler();
    
    void start_upload_timing();
    void end_upload_timing();
    
    void start_compute_timing();
    void end_compute_timing();
    
    void start_download_timing();
    void end_download_timing();
    
    BasicProfilingResult get_results(const std::string& algorithm, const StrobeConfig& config);
};

} // namespace CudaCalc
```

#### Вариант 2: MemoryProfiler (опционально) - Расширенный

**memory_profiler.h**
```cpp
#pragma once
#include <cuda_runtime.h>
#include <string>

namespace CudaCalc {

// Расширенное профилирование памяти и GPU
struct MemoryProfilingResult {
    // Memory usage
    size_t allocated_vram_mb;     // Выделенная VRAM
    size_t peak_vram_mb;          // Пиковая VRAM
    float memory_bandwidth_gbps;  // Memory bandwidth
    
    // GPU utilization
    float gpu_utilization;        // GPU utilization %
    float occupancy;              // Occupancy %
    float warp_efficiency;        // Warp efficiency %
    
    // Metadata
    std::string algorithm;
    std::string timestamp;
};

class MemoryProfiler {
private:
    size_t initial_free_mem_;
    size_t initial_total_mem_;
    
public:
    MemoryProfiler();
    ~MemoryProfiler();
    
    void start_monitoring();
    void end_monitoring();
    
    MemoryProfilingResult get_results(const std::string& algorithm);
    
    // Утилиты
    size_t get_free_memory() const;
    size_t get_total_memory() const;
    float get_memory_bandwidth() const;
};

} // namespace CudaCalc
```

#### Полный результат профилирования

**profiling_data.h**
```cpp
#pragma once
#include "basic_profiler.h"
#include "memory_profiler.h"

namespace CudaCalc {

// Полный пакет профилирования
struct FullProfilingResult {
    BasicProfilingResult basic;      // Обязательно (CUDA Events)
    MemoryProfilingResult memory;    // Опционально (Memory + GPU)
    bool has_memory_profiling;       // Флаг наличия расширенного профилирования
};

} // namespace CudaCalc
```

### 4.5 FFT16 Реализации

#### Вариант A: Tensor Cores (wmma)

**fft16_wmma.cu:**
```cpp
__global__ void fft16_wmma_kernel(
    const cuComplex* input,
    cuComplex* output,
    int num_windows
) {
    // 64 FFT в одном блоке
    int block_fft_id = threadIdx.x / 16;  // 0-63
    int point_id = threadIdx.x % 16;       // 0-15
    int global_fft_id = blockIdx.x * 64 + block_fft_id;
    
    if (global_fft_id >= num_windows) return;
    
    // Shared memory: [64 FFTs][16 points]
    __shared__ __half2 shared_data[64][16];  // FP16 комплексные
    
    // Load input в FP16
    int input_idx = global_fft_id * 16 + point_id;
    shared_data[block_fft_id][point_id] = __floats2half2_rn(input[input_idx].x, input[input_idx].y);
    
    __syncthreads();
    
    // ============= ЛИНЕЙНАЯ РАСКРУТКА 4 STAGES =============
    
    // STAGE 0: step = 1, group_size = 2
    {
        int idx1 = (point_id / 2) * 2 + (point_id % 1);
        int idx2 = idx1 + 1;
        // ... butterfly operation ...
    }
    __syncthreads();
    
    // STAGE 1: step = 2, group_size = 4
    {
        int idx1 = (point_id / 2) * 4 + (point_id % 2);
        int idx2 = idx1 + 2;
        // ... butterfly operation ...
    }
    __syncthreads();
    
    // STAGE 2: step = 4, group_size = 8
    {
        int idx1 = (point_id / 4) * 8 + (point_id % 4);
        int idx2 = idx1 + 4;
        // ... butterfly operation ...
    }
    __syncthreads();
    
    // STAGE 3: step = 8, group_size = 16
    {
        int idx1 = (point_id / 8) * 16 + (point_id % 8);
        int idx2 = idx1 + 8;
        // ... butterfly operation ...
    }
    __syncthreads();
    
    // ============= FFT SHIFT (в kernel) =============
    // Порядок: [-f8, -f7, ..., -f1, DC, f1, ..., f7]
    int output_idx_shifted;
    if (point_id < 8) {
        output_idx_shifted = point_id + 8;  // DC, f1, ..., f7 → positions 8-15
    } else {
        output_idx_shifted = point_id - 8;  // f8, -f7, ..., -f1 → positions 0-7
    }
    
    int output_idx = global_fft_id * 16 + output_idx_shifted;
    __half2 result = shared_data[block_fft_id][point_id];
    output[output_idx].x = __low2float(result);
    output[output_idx].y = __high2float(result);
}
```

#### Вариант B: 2D Shared Memory (FP32)

**fft16_shared2d.cu:**
```cpp
__global__ void fft16_shared2d_kernel(
    const cuComplex* input,
    cuComplex* output,
    int num_windows
) {
    int block_fft_id = threadIdx.x / 16;
    int point_id = threadIdx.x % 16;
    int global_fft_id = blockIdx.x * 64 + block_fft_id;
    
    if (global_fft_id >= num_windows) return;
    
    // 2D shared memory: [64 FFTs][16 points]
    __shared__ float2 shared_data[64][16];  // FP32
    
    // Load
    int input_idx = global_fft_id * 16 + point_id;
    shared_data[block_fft_id][point_id] = make_float2(input[input_idx].x, input[input_idx].y);
    
    __syncthreads();
    
    // Линейная раскрутка 4 stages (аналогично wmma)
    // ... STAGE 0 ...
    // ... STAGE 1 ...
    // ... STAGE 2 ...
    // ... STAGE 3 ...
    
    // FFT shift и запись
    // ... аналогично wmma ...
}
```

### 4.6 JSON логирование

**Формат JSON:**
```json
{
  "test_info": {
    "date": "2025-10-09",
    "time": "10:30:45",
    "gpu": "NVIDIA RTX 3060",
    "cuda_version": "13.0",
    "driver_version": "535.104.05",
    "compute_capability": "8.6"
  },
  "test_config": {
    "ray_count": 4,
    "points_per_ray": 1024,
    "total_points": 4096,
    "window_fft": 16,
    "num_windows": 256,
    "signal_type": "SINE",
    "sine_period": 8,
    "amplitude": 1.0,
    "phase": 0.0
  },
  "results": [
    {
      "algorithm": "FFT16_WMMA",
      "profiling": {
        "basic": {
          "upload_ms": 0.123,
          "compute_ms": 0.456,
          "download_ms": 0.089,
          "total_ms": 0.668
        },
        "memory": {
          "enabled": true,
          "allocated_vram_mb": 32,
          "peak_vram_mb": 45,
          "memory_bandwidth_gbps": 450,
          "gpu_utilization": 92.5,
          "occupancy": 87.3,
          "warp_efficiency": 89.1
        }
      },
      "validation": {
        "enabled": true,
        "max_error": 1.23e-6,
        "mean_error": 3.45e-7,
        "tolerance": 0.0001,
        "passed": true
      }
    },
    {
      "algorithm": "FFT16_Shared2D",
      "profiling": {
        "basic": {
          "upload_ms": 0.125,
          "compute_ms": 0.512,
          "download_ms": 0.091,
          "total_ms": 0.728
        },
        "memory": {
          "enabled": true,
          "allocated_vram_mb": 28,
          "peak_vram_mb": 38,
          "memory_bandwidth_gbps": 420,
          "gpu_utilization": 88.2,
          "occupancy": 82.1,
          "warp_efficiency": 85.7
        }
      },
      "validation": {
        "enabled": true,
        "max_error": 4.56e-7,
        "mean_error": 1.23e-7,
        "tolerance": 0.0001,
        "passed": true
      }
    }
  ],
  "conclusion": {
    "fastest_algorithm": "FFT16_WMMA",
    "fastest_time_ms": 0.668,
    "speedup": 1.09
  }
}
```

---

## 5. Workflow выполнения теста

```
┌─────────────────┐
│ 1. MainProgram  │
│  main_fft16.cpp │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ 2. SignalGenerators     │
│  SineGenerator          │
│  - Генерация 4096 точек │
│  - Строб: 4×1024        │
│  - Период синуса = 8    │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ 3. DataContext                           │
│  🔴 КРИТИЧНО: Генерация ValidationData   │
│  ОДИН РАЗ через cuFFT:                   │
│  - cufftExecC2C() на весь сигнал         │
│  - Разбивка на 256 окон × 16 гармоник    │
│  - Сохранение в ValidationData.reference │
│  - Если is_validate = false → nullptr    │
│  Формирование TestDataPackage            │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 4. Tester                    │
│  GPUProfiler::start()        │
└──────────┬───────────────────┘
           │
           ├─────────────────────────────┐
           │                             │
           ▼                             ▼
┌────────────────────────┐    ┌────────────────────────┐
│ 5a. FFT16_WMMA         │    │ 5b. FFT16_Shared2D     │
│  - cudaMemcpy H→D      │    │  - cudaMemcpy H→D      │
│    [profile upload]    │    │    [profile upload]    │
│  - kernel<<<4, 1024>>> │    │  - kernel<<<4, 1024>>> │
│    [profile compute]   │    │    [profile compute]   │
│  - cudaMemcpy D→H      │    │  - cudaMemcpy D→H      │
│    [profile download]  │    │    [profile download]  │
└────────────┬───────────┘    └────────────┬───────────┘
             │                             │
             ▼                             ▼
┌────────────────────────────────────────────┐
│ 6. Validation                              │
│  FFTValidator::validate()                  │
│  - Сравнение с cuFFT                       │
│  - Вычисление ошибок                       │
│  - Проверка tolerance                      │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│ 7. DataContext                             │
│  JSONLogger::write()                       │
│  - Сохранение всех результатов             │
│  - Определение fastest algorithm           │
│                                            │
│  🔴 ОБЯЗАТЕЛЬНО: ModelArchiver::save()     │
│  - Сохранение исходников (.cu, .cpp)      │
│  - Сохранение results.json                │
│  - Сохранение validation.json             │
│  - Версионирование (v1, v2, v3, ...)      │
│  - Путь: Models/NVIDIA/FFT/16/model_...   │
└────────────────────────────────────────────┘
```

---

## 6. Критерии приёмки

### 6.1 Тестирование
- [ ] Unit тесты для генератора синусоид
- [ ] Проверка корректности FFT16 через cuFFT
- [ ] Сравнение двух реализаций (wmma vs shared2d)
- [ ] Проверка fftshift (порядок гармоник)
- [ ] Профилирование работает корректно
- [ ] JSON валидный и содержит все поля
- [ ] 🔴 **ModelArchiver сохраняет модели без потерь**
- [ ] 🔴 **Версионирование работает (v1, v2, v3, ...)**
- [ ] 🔴 **Исходники сохраняются в Models/**

### 6.2 Производительность
- [ ] Compute time FFT16 измерен
- [ ] Upload/Download time измерен
- [ ] Определен fastest algorithm
- [ ] Baseline метрики задокументированы

### 6.3 Качество кода
- [ ] Code review пройден
- [ ] CMake собирается на Ubuntu
- [ ] Нет memory leaks (cuda-memcheck)
- [ ] Код соответствует constitution.md

### 6.4 Документация
- [ ] Этот spec.md заполнен
- [ ] CLAUDE.md обновлен с новой фичей
- [ ] Примеры использования в quickstart.md

---

## 7. Зависимости

### 7.1 Внешние библиотеки
- CUDA Toolkit 13.x (cuFFT, cudart)
- nlohmann/json (для JSON логирования)
- Google Test (опционально, для unit тестов)

### 7.2 Внутренние модули
- Interface (базовые определения)
- SignalGenerators (новый модуль)
- DataContext
- ModelsFunction
- Tester

---

## 8. Риски и митигация

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Tensor Cores медленнее обычной реализации | Средняя | Высокое | Поэтому делаем ОБА варианта и сравниваем |
| FP16 потеря точности | Высокая | Среднее | Валидация покажет, приемлемо ли |
| Сложность реализации wmma | Высокая | Среднее | Изучаем существующий код, используем примеры |
| Проблемы с CMake на Ubuntu | Низкая | Высокое | Тестируем сразу, используем стандартные пути |

---

## 9. План реализации (фазы)

### Фаза 1: Базовая инфраструктура (2-3 дня)
- [ ] Настроить CMake для Ubuntu
- [ ] Создать структуру модулей
- [ ] Определить интерфейсы (Interface/)
- [ ] 🔴 **Создать директории Models/NVIDIA/FFT/16/**

### Фаза 2: Генератор сигналов + Валидация (1-2 дня)
- [ ] SignalGenerators/ модуль
- [ ] SineGenerator реализация
- [ ] **DataContext: генерация ValidationData через cuFFT** 
- [ ] Unit тесты

### Фаза 3: FFT16 реализации (3-4 дня)
- [ ] FFT16 Shared2D (проще, начинаем с него)
- [ ] FFT16 WMMA (сложнее)
- [ ] Линейная раскрутка для обеих
- [ ] fftshift в kernel

### Фаза 4: Тестирование (2-3 дня)
- [ ] GPUProfiler (cudaEvent)
- [ ] FFTValidator (cuFFT)
- [ ] JSON logger
- [ ] Интеграция всей цепочки

### Фаза 5: 🔴 ОБЯЗАТЕЛЬНО! ModelArchiver (1-2 дня)
- [ ] **ModelArchiver класс**
- [ ] **Сохранение исходников (.cu, .cpp)**
- [ ] **Автоматическое версионирование**
- [ ] **Функции load/compare/list моделей**
- [ ] **Интеграция в workflow**

### Фаза 6: Оптимизация и финал (1-2 дня)
- [ ] Сравнение производительности WMMA vs Shared2D
- [ ] Выбор лучшего варианта
- [ ] Сохранение лучшей модели через ModelArchiver
- [ ] Документирование результатов

**Итого:** ~12 дней разработки (с ModelArchiver)

---

## 10. Следующие шаги

После завершения этой спецификации:

1. Создать `plan.md` с детальным планом реализации
2. Создать `tasks.md` с разбивкой на задачи
3. Настроить CMake
4. Начать реализацию по фазам

---

## История изменений

| Дата | Версия | Автор | Изменения |
|------|--------|-------|-----------|
| 2025-10-09 | 1.0 | AlexLan73 | Создание спецификации FFT16 baseline pipeline |

---

**Статус:** ✅ Готов к review и созданию плана реализации

