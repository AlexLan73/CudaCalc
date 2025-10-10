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
2. **Три реализации FFT16** для сравнения скорости:
   - Вариант A: Tensor Cores (wmma) с FP16
   - Вариант B: Обычный 2D shared memory с FP32
   - Вариант C: cuFFT wrapper (эталон производительности)
3. **Профилирование** через CUDA Events (upload, compute, download)
4. **Валидация** через Python (NumPy/SciPy) - отдельный скрипт
5. **JSON логирование** с версионированием файлов по дате
6. **Визуализация** результатов (сигнал, огибающая, спектры)

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

**FR-2: Три реализации FFT16**
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
- **Вариант C - cuFFT Wrapper:**
  - Обертка над cuFFT (batch FFT)
  - **Цель:** Эталон производительности для сравнения
  - **НЕ для валидации** (валидация в Python)
  - Профилирование через CUDA Events (как для A и B)
  - Результаты в JSON: `reference_cufft_ms`
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

**FR-4: Валидация результатов через Python**
- **АРХИТЕКТУРНОЕ РЕШЕНИЕ:** Валидация вынесена в Python (отдельно от C++ Tester)
- **Reference реализация:** NumPy/SciPy FFT
- **Управление:** флаг `return_for_validation` в `InputSignalData`
- **Если `return_for_validation == true`:**
  - C++ Tester возвращает данные в DataContext
  - DataContext сохраняет JSON с входным сигналом + результатами GPU
  - Формат файла: `YYYY-MM-DD_HH-MM_<algorithm>_test.json`
  - **Версионирование:** Старые файлы НЕ перезаписываются
- **Если `return_for_validation == false`:**
  - Данные не сохраняются (только профилирование)
- **Python валидатор:**
  - Отдельный скрипт `validate_fft.py`
  - По умолчанию: читает последний файл (по дате)
  - Можно указать конкретный файл
  - Вычисляет FFT через scipy.fft
  - Сравнивает с GPU результатами
  - Выводит таблицу результатов в консоль
  - **Визуализация (matplotlib):**
    - Входной сигнал (комплексный)
    - Огибающая сигнала
    - Спектры: GPU vs Python (для сравнения)
    - Два режима: с графиками / без графиков
  - Метрика: относительная ошибка < 0.01%
- Приоритет: **Критический**

**FR-5: JSON логирование с версионированием**
- **Профилирование:** `DataContext/Reports/YYYY-MM-DD_HH-MM_profiling.json`
- **Данные валидации:** `DataContext/ValidationData/FFT16/YYYY-MM-DD_HH-MM_<algorithm>_test.json`
- **Описание в имени файла:**
  - Автоматически: из конфига или `<algorithm>_test`
  - Примеры: `2025-10-09_14-30_fft16_wmma_test.json`
- **Версионирование:** Старые файлы НЕ перезаписываются
- **Формат JSON:** см. раздел 4.6
- Приоритет: **Высокий**

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
│   │   ├── signal_data.h          # Структуры данных сигналов (ОБНОВЛЕНО)
│   │   └── spectral_data.h        # Структуры спектральных данных
│   └── CMakeLists.txt
│
├── SignalGenerators/               # Генераторы сигналов
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
│   │   ├── json_logger.h          # JSON логирование (ОБНОВЛЕНО)
│   │   └── model_archiver.h       # 🔴 Архивирование моделей
│   ├── src/
│   │   ├── data_manager.cpp
│   │   ├── json_logger.cpp
│   │   └── model_archiver.cpp
│   ├── Config/                     # ⭐ НОВЫЙ: Конфигурационные файлы
│   │   ├── paths.json             # Пути к каталогам
│   │   ├── validation_params.json # Параметры валидации
│   │   └── samples/               # Образцы данных
│   ├── Reports/                    # JSON отчёты профилирования
│   ├── ValidationData/             # ⭐ НОВЫЙ: Данные для Python валидации
│   │   └── FFT16/                 # Отдельно для каждого размера FFT
│   │       ├── 2025-10-09_10-30_fft16_wmma_test.json
│   │       ├── 2025-10-09_14-15_fft16_shared2d_test.json
│   │       └── ...                # Версионирование по дате
│   ├── Models/                     # 🔴 Архив моделей и результатов
│   │   └── NVIDIA/
│   │       └── FFT/
│   │           └── 16/
│   │               ├── model_2025_10_09_v1/
│   │               │   ├── fft16_wmma.cu
│   │               │   ├── fft16_wmma.cpp
│   │               │   ├── description.txt
│   │               │   ├── results.json
│   │               │   └── validation.json
│   │               └── model_2025_10_09_v2/
│   └── CMakeLists.txt
│
├── ModelsFunction/                 # Экспериментальные модели
│   ├── include/
│   │   └── nvidia/
│   │       └── fft/
│   │           ├── fft16_wmma.h       # Tensor Cores версия
│   │           ├── fft16_shared2d.h   # 2D shared memory версия
│   │           └── fft16_cufft.h      # ⭐ НОВЫЙ: cuFFT wrapper
│   ├── src/
│   │   └── nvidia/
│   │       └── fft/
│   │           ├── fft16_wmma.cu      # Kernel wmma
│   │           ├── fft16_wmma.cpp     # Wrapper
│   │           ├── fft16_shared2d.cu  # Kernel 2D
│   │           ├── fft16_shared2d.cpp # Wrapper
│   │           └── fft16_cufft.cpp    # ⭐ НОВЫЙ: cuFFT wrapper
│   └── CMakeLists.txt
│
├── Tester/                         # Система тестирования
│   ├── include/
│   │   └── performance/
│   │       ├── basic_profiler.h       # Базовое профилирование (CUDA Events)
│   │       ├── memory_profiler.h      # Расширенное (Memory + GPU)
│   │       └── profiling_data.h       # Структуры данных профайлинга
│   ├── src/
│   │   └── performance/
│   │       ├── basic_profiler.cpp     # Обязательно (baseline)
│   │       └── memory_profiler.cpp    # Опционально (расширенный)
│   └── CMakeLists.txt
│   # ⚠️ УДАЛЕНО: validation/ (валидация теперь в Python)
│
├── Validator/                      # ⭐ НОВЫЙ: Python валидатор
│   ├── validate_fft.py            # Главный скрипт валидации
│   ├── fft_reference.py           # Эталонные вычисления (scipy)
│   ├── comparison.py              # Сравнение результатов
│   ├── visualization.py           # Визуализация (matplotlib)
│   ├── requirements.txt           # numpy, scipy, matplotlib
│   └── README.md                  # Инструкции по использованию
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
    
    bool return_for_validation;               // ⭐ НОВОЕ: Возвращать ли данные для валидации
                                              // true  = сохранить JSON для Python
                                              // false = только профилирование
    
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

// ⚠️ ИЗМЕНЕНО: ValidationData теперь не нужен в C++
// Валидация полностью перенесена в Python

// Полный пакет данных для теста (УПРОЩЕНО)
struct TestDataPackage {
    InputSignalData input;
    // ValidationData УДАЛЕНО - валидация в Python
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

#### 4.6.1 JSON Профилирования (Reports/)

**Путь:** `DataContext/Reports/YYYY-MM-DD_HH-MM_profiling.json`

**Формат:**
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
          "enabled": false
        }
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
          "enabled": false
        }
      }
    },
    {
      "algorithm": "cuFFT_Reference",
      "profiling": {
        "basic": {
          "upload_ms": 0.120,
          "compute_ms": 0.380,
          "download_ms": 0.088,
          "total_ms": 0.588
        },
        "memory": {
          "enabled": false
        }
      }
    }
  ],
  "conclusion": {
    "fastest_algorithm": "cuFFT_Reference",
    "fastest_time_ms": 0.588,
    "custom_algorithms": ["FFT16_WMMA", "FFT16_Shared2D"],
    "speedup_vs_cufft": {
      "FFT16_WMMA": 0.88,
      "FFT16_Shared2D": 0.81
    }
  }
}
```

#### 4.6.2 JSON Данных валидации (ValidationData/)

**Путь:** `DataContext/ValidationData/FFT16/YYYY-MM-DD_HH-MM_<algorithm>_test.json`

**Примеры имен файлов:**
- `2025-10-09_10-30_fft16_wmma_test.json`
- `2025-10-09_14-15_fft16_shared2d_test.json`
- `2025-10-10_09-00_fft16_wmma_optimized_v2.json`

**Формат:**
```json
{
  "metadata": {
    "date": "2025-10-09",
    "time": "10:30:45",
    "gpu_model": "NVIDIA RTX 3060",
    "cuda_version": "13.0",
    "driver_version": "535.104.05",
    "algorithm": "FFT16_WMMA",
    "description": "fft16_wmma_test"
  },
  "test_config": {
    "ray_count": 4,
    "points_per_ray": 1024,
    "window_fft": 16,
    "signal_type": "SINE",
    "sine_period": 8,
    "amplitude": 1.0,
    "phase": 0.0
  },
  "input_signal": {
    "real": [1.0, 0.707, 0.0, -0.707, ...],
    "imag": [0.0, 0.707, 1.0, 0.707, ...]
  },
  "gpu_results": {
    "num_windows": 256,
    "windows": [
      {
        "window_id": 0,
        "spectrum_real": [0.0, 1.5, 0.3, ...],
        "spectrum_imag": [0.0, -0.2, 0.8, ...]
      },
      {
        "window_id": 1,
        "spectrum_real": [0.0, 1.6, 0.2, ...],
        "spectrum_imag": [0.0, -0.3, 0.9, ...]
      }
    ]
  }
}
```

**⚠️ Примечание:** Validation результаты (max_error, passed) теперь НЕ в C++ JSON, 
а вычисляются Python скриптом!

### 4.7 Python Validator (Validator/)

#### 4.7.1 Структура Python модуля

```
Validator/
├── validate_fft.py        # Главный скрипт валидации
├── fft_reference.py       # Эталонные вычисления (scipy.fft)
├── comparison.py          # Сравнение результатов
├── visualization.py       # Визуализация (matplotlib)
├── requirements.txt       # Зависимости
└── README.md              # Инструкции
```

#### 4.7.2 Использование

**Базовое использование:**
```bash
# По умолчанию: читает последний файл из ValidationData/FFT16/
python validate_fft.py

# Указать конкретный файл
python validate_fft.py --file "2025-10-09_10-30_fft16_wmma_test.json"

# С визуализацией
python validate_fft.py --visualize

# Без графиков (только таблица)
python validate_fft.py --no-plot
```

#### 4.7.3 Вывод в консоль

```
=== FFT Validation Report ===
File: 2025-10-09_10-30_fft16_wmma_test.json
GPU: NVIDIA RTX 3060
Algorithm: FFT16_WMMA
Date: 2025-10-09 10:30:45

Input Signal: 4096 points (4 rays × 1024 points)
FFT Windows: 256 windows × 16 points

Reference: scipy.fft.fft (NumPy)

Comparison Results:
┌─────────┬──────────────┬──────────────┬────────┐
│ Window  │ Max Error    │ Mean Error   │ Status │
├─────────┼──────────────┼──────────────┼────────┤
│ 0       │ 1.23e-06     │ 3.45e-07     │ PASS   │
│ 1       │ 2.34e-06     │ 4.56e-07     │ PASS   │
│ ...     │ ...          │ ...          │ ...    │
│ 255     │ 1.87e-06     │ 2.98e-07     │ PASS   │
└─────────┴──────────────┴──────────────┴────────┘

Overall Statistics:
  Max Error (all windows):  2.34e-06
  Mean Error (all windows): 3.12e-07
  Tolerance:                1.00e-04
  Passed Windows:           256/256 (100%)

✅ VALIDATION PASSED
```

#### 4.7.4 Визуализация (matplotlib)

**Режимы визуализации:**

1. **С графиками (`--visualize`):**
   - Выводит 3 subplot'а:
     ```
     [Subplot 1: Входной сигнал]
     - Real part (синяя линия)
     - Imaginary part (красная линия)
     - Огибающая |signal| (зеленая линия, пунктир)
     
     [Subplot 2: Спектр GPU (первое окно)]
     - Magnitude |spectrum| (столбцы)
     - Подписи частотных бинов
     
     [Subplot 3: Сравнение GPU vs Python]
     - GPU результат (синие столбцы)
     - Python результат (красные крестики)
     - Разница (зеленая линия)
     ```

2. **Без графиков (`--no-plot` по умолчанию):**
   - Только таблица в консоль
   - Быстрая валидация

#### 4.7.5 Модуль fft_reference.py

```python
import numpy as np
from scipy import fft

def compute_reference_fft(signal, window_size=16):
    """
    Вычисляет эталонный FFT через scipy
    
    Args:
        signal: комплексный массив (4096 точек)
        window_size: размер окна FFT (16)
    
    Returns:
        windows: list из 256 массивов (каждый 16 точек)
    """
    num_windows = len(signal) // window_size
    windows = []
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window_data = signal[start:end]
        
        # FFT через scipy
        spectrum = fft.fft(window_data)
        
        # fftshift (как в GPU kernel)
        spectrum_shifted = fft.fftshift(spectrum)
        
        windows.append(spectrum_shifted)
    
    return windows
```

#### 4.7.6 Модуль comparison.py

```python
import numpy as np

def compare_results(gpu_windows, reference_windows, tolerance=1e-4):
    """
    Сравнивает GPU результаты с эталонными
    
    Args:
        gpu_windows: list из 256 массивов (GPU)
        reference_windows: list из 256 массивов (scipy)
        tolerance: допустимая относительная ошибка
    
    Returns:
        dict с результатами валидации
    """
    errors = []
    
    for gpu, ref in zip(gpu_windows, reference_windows):
        # Относительная ошибка
        diff = np.abs(gpu - ref)
        ref_mag = np.abs(ref)
        
        # Избегаем деления на ноль
        rel_error = np.where(ref_mag > 1e-10, diff / ref_mag, diff)
        
        max_error = np.max(rel_error)
        mean_error = np.mean(rel_error)
        
        errors.append({
            'max': max_error,
            'mean': mean_error,
            'passed': max_error < tolerance
        })
    
    return {
        'per_window': errors,
        'overall_max': max([e['max'] for e in errors]),
        'overall_mean': np.mean([e['mean'] for e in errors]),
        'tolerance': tolerance,
        'passed_windows': sum([e['passed'] for e in errors]),
        'total_windows': len(errors)
    }
```

#### 4.7.7 Модуль visualization.py

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(signal, gpu_spectrum, ref_spectrum, window_id=0):
    """
    Визуализация входного сигнала и спектров
    
    Args:
        signal: входной комплексный сигнал (4096 точек)
        gpu_spectrum: GPU результат (256 окон × 16 точек)
        ref_spectrum: Reference результат (256 окон × 16 точек)
        window_id: какое окно показать (по умолчанию 0)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Subplot 1: Входной сигнал
    ax1 = axes[0]
    t = np.arange(len(signal))
    ax1.plot(t, signal.real, 'b-', label='Real', linewidth=1)
    ax1.plot(t, signal.imag, 'r-', label='Imaginary', linewidth=1)
    ax1.plot(t, np.abs(signal), 'g--', label='Envelope |signal|', linewidth=1.5)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Input Signal (4096 points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: GPU спектр
    ax2 = axes[1]
    gpu_mag = np.abs(gpu_spectrum[window_id])
    freq_bins = np.arange(16) - 8  # fftshift: [-8, -7, ..., 7]
    ax2.bar(freq_bins, gpu_mag, color='blue', alpha=0.7, label='GPU FFT')
    ax2.set_xlabel('Frequency bin')
    ax2.set_ylabel('Magnitude')
    ax2.set_title(f'GPU Spectrum (Window {window_id})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Сравнение
    ax3 = axes[2]
    ref_mag = np.abs(ref_spectrum[window_id])
    diff = np.abs(gpu_mag - ref_mag)
    
    ax3.bar(freq_bins, gpu_mag, color='blue', alpha=0.5, label='GPU')
    ax3.plot(freq_bins, ref_mag, 'rx', markersize=10, label='Python (scipy)', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(freq_bins, diff, 'g-', label='Difference', linewidth=1.5)
    
    ax3.set_xlabel('Frequency bin')
    ax3.set_ylabel('Magnitude')
    ax3_twin.set_ylabel('|GPU - Python|', color='g')
    ax3.set_title('GPU vs Python Comparison')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

#### 4.7.8 requirements.txt

```
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

#### 4.7.9 Установка (Windows)

```powershell
# Создать виртуальное окружение
python -m venv venv

# Активировать
.\venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

#### 4.7.10 Установка (Ubuntu)

```bash
# Установить Python 3 и pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Создать виртуальное окружение
python3 -m venv venv

# Активировать
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
```

---

## 5. Workflow выполнения теста

### 5.1 C++ Часть (Профилирование)

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
│  - return_for_validation│
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 3. DataContext               │
│  Формирование TestDataPackage│
│  (БЕЗ ValidationData!)       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 4. Tester                    │
│  BasicProfiler::start()      │
└──────────┬───────────────────┘
           │
           ├──────────────────────────────────┐
           │                                  │
           ▼                                  ▼
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
             └──────────┬──────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │ 5c. cuFFT Wrapper      │
           │  - cudaMemcpy H→D      │
           │    [profile upload]    │
           │  - cufftExecC2C()      │
           │    [profile compute]   │
           │  - cudaMemcpy D→H      │
           │    [profile download]  │
           └────────────┬───────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│ 6. DataContext - Сохранение                │
│                                            │
│ A) Профилирование (Reports/):              │
│    JSONLogger::write_profiling()           │
│    → YYYY-MM-DD_HH-MM_profiling.json       │
│    - Результаты всех 3 алгоритмов          │
│    - Fastest algorithm                     │
│    - Speedup vs cuFFT                      │
│                                            │
│ B) Данные валидации (ValidationData/):     │
│    ЕСЛИ return_for_validation == true:     │
│    JSONLogger::write_validation_data()     │
│    → ValidationData/FFT16/                 │
│       YYYY-MM-DD_HH-MM_<algo>_test.json    │
│    - Входной сигнал (input_signal)         │
│    - Результаты GPU (gpu_results)          │
│    - Метаданные (metadata)                 │
│                                            │
│ C) 🔴 Архивирование (Models/):             │
│    ModelArchiver::save()                   │
│    → Models/NVIDIA/FFT/16/model_vN/        │
│    - Исходники (.cu, .cpp)                 │
│    - results.json                          │
│    - description.txt                       │
└────────────────────────────────────────────┘
```

### 5.2 Python Часть (Валидация) - Запуск вручную

```
┌─────────────────────────────────────────┐
│ 7. Python Validator (РУЧНОЙ ЗАПУСК)    │
│  python validate_fft.py                 │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 8. Чтение ValidationData                │
│  - По умолчанию: последний файл         │
│  - ValidationData/FFT16/                │
│    YYYY-MM-DD_HH-MM_<algo>_test.json    │
│  - Парсинг input_signal + gpu_results   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 9. Эталонные вычисления                │
│  fft_reference.py                       │
│  - scipy.fft.fft() для каждого окна     │
│  - 256 окон × 16 точек                  │
│  - fftshift                             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 10. Сравнение результатов               │
│  comparison.py                          │
│  - Относительная ошибка для каждого окна│
│  - max_error, mean_error                │
│  - Проверка tolerance (< 0.01%)         │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 11. Вывод результатов                   │
│  A) Таблица в консоль:                  │
│     - Window │ Max Error │ Status       │
│     - Overall statistics                │
│     - PASS / FAIL                       │
│                                         │
│  B) Визуализация (если --visualize):    │
│     visualization.py                    │
│     - Subplot 1: Входной сигнал         │
│     - Subplot 2: GPU спектр             │
│     - Subplot 3: GPU vs Python          │
│     - matplotlib.show()                 │
└─────────────────────────────────────────┘
```

---

## 6. Критерии приёмки

### 6.1 Тестирование
- [ ] Unit тесты для генератора синусоид
- [ ] Проверка корректности FFT16 через **Python валидатор** (scipy)
- [ ] Сравнение **трёх** реализаций (wmma vs shared2d vs cuFFT)
- [ ] Проверка fftshift (порядок гармоник)
- [ ] Профилирование работает корректно (upload/compute/download)
- [ ] JSON профилирования валидный (Reports/)
- [ ] JSON валидационных данных валидный (ValidationData/)
- [ ] **Версионирование файлов по дате** (не перезаписываются)
- [ ] Python валидатор работает корректно
- [ ] Визуализация matplotlib работает (--visualize)
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
- [ ] 🔴 **Создать директории:**
  - [ ] `DataContext/Config/`
  - [ ] `DataContext/ValidationData/FFT16/`
  - [ ] `DataContext/Models/NVIDIA/FFT/16/`
  - [ ] `Validator/`

### Фаза 2: Генератор сигналов (1 день)
- [ ] SignalGenerators/ модуль
- [ ] SineGenerator реализация
- [ ] **Обновить InputSignalData: добавить return_for_validation**
- [ ] Unit тесты

### Фаза 3: FFT16 реализации (4-5 дней)
- [ ] FFT16 Shared2D (проще, начинаем с него)
- [ ] FFT16 WMMA (сложнее)
- [ ] **FFT16 cuFFT Wrapper** (эталон производительности)
- [ ] Линейная раскрутка для WMMA и Shared2D
- [ ] fftshift в kernel

### Фаза 4: Профилирование (2 дня)
- [ ] BasicProfiler (cudaEvent)
- [ ] JSON logger для профилирования (Reports/)
- [ ] JSON logger для валидационных данных (ValidationData/)
- [ ] **Версионирование файлов по дате**
- [ ] Интеграция всей цепочки

### Фаза 5: Python Validator (2-3 дня)
- [ ] **Python окружение: создать venv**
- [ ] **requirements.txt** (numpy, scipy, matplotlib)
- [ ] **Инструкции установки для Ubuntu**
- [ ] **fft_reference.py** (scipy.fft)
- [ ] **comparison.py** (сравнение результатов)
- [ ] **visualization.py** (matplotlib, 3 subplot'а)
- [ ] **validate_fft.py** (главный скрипт)
- [ ] **Тестирование валидатора**

### Фаза 6: 🔴 ModelArchiver (1-2 дня)
- [ ] **ModelArchiver класс**
- [ ] **Сохранение исходников (.cu, .cpp)**
- [ ] **Автоматическое версионирование**
- [ ] **Функции load/compare/list моделей**
- [ ] **Интеграция в workflow**

### Фаза 7: Оптимизация и финал (1-2 дня)
- [ ] Сравнение производительности: WMMA vs Shared2D vs cuFFT
- [ ] Валидация корректности через Python
- [ ] Визуализация результатов
- [ ] Выбор лучшего варианта
- [ ] Сохранение лучшей модели через ModelArchiver
- [ ] Документирование результатов

**Итого:** ~14-17 дней разработки (с Python валидатором)

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

