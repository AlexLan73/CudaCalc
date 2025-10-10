# План реализации: FFT16 Baseline Testing Pipeline

**Автор:** AlexLan73  
**Дата:** 10 октября 2025  
**Статус:** In Progress  
**Основано на:** [spec.md](spec.md) v1.2

---

## 1. Обзор плана

### 1.1 Цель
Реализовать **полную тестовую цепочку** для FFT16 с двумя вариантами реализации, профилированием, валидацией и архивированием результатов.

### 1.2 Scope
- Базовая архитектура (5 модулей)
- Генератор синусоидальных сигналов
- Две реализации FFT16 (Tensor Cores vs 2D Shared)
- Профилирование (Basic + опционально Memory)
- Валидация через Python (сравнение с scipy.fft)
- Система архивирования моделей (ModelArchiver)
- CMake build system для Ubuntu

### 1.3 Временные рамки
**Общий срок:** 10-12 рабочих дней (~2 недели)

---

## 2. Архитектура системы

### 2.1 Диаграмма модулей

```
┌─────────────────────────────────────────────────────────────┐
│                     MainProgram                             │
│                  main_fft16_test.cpp                        │
│                                                             │
│  Workflow:                                                  │
│  1. Создание конфигурации                                   │
│  2. Генерация сигнала                                       │
│  3. Создание TestDataPackage                                │
│  4. Запуск тестов (WMMA + Shared2D)                         │
│  5. Сохранение результатов                                  │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├──────────────────┬──────────────────┬────────────┐
           │                  │                  │            │
           ▼                  ▼                  ▼            ▼
    ┌────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐
    │ Interface/ │   │ SignalGen/   │   │ Models/  │   │ Tester/  │
    │            │   │              │   │ Function │   │          │
    │ - Базовые  │   │ - Sine       │   │          │   │ - Profile│
    │   интерфейсы│   │   Generator  │   │ - FFT16  │   │ - Validate│
    │ - Структуры│   │ - Enum типов │   │   WMMA   │   │ - JSON   │
    │   данных   │   │   сигналов   │   │ - FFT16  │   │          │
    │            │   │              │   │   Shared │   │          │
    └────────────┘   └──────────────┘   └──────────┘   └──────────┘
           │                  │                  │            │
           └──────────────────┴──────────────────┴────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ DataContext/ │
                     │              │
                     │ - Data Mgr   │
                     │ - JSON Log   │
                     │ - Model      │
                     │   Archiver   │
                     │ - Models/    │
                     │ - Reports/   │
                     │ - Validation │
                     │   Data/      │
                     └──────────────┘
```

### 2.2 Зависимости модулей

```
CMakeLists.txt (ROOT)
    ├─> Interface (no dependencies)
    ├─> SignalGenerators
    │   └─> depends: Interface
    ├─> DataContext
    │   └─> depends: Interface, SignalGenerators
    ├─> ModelsFunction
    │   └─> depends: Interface
    ├─> Tester
    │   └─> depends: Interface, DataContext
    └─> MainProgram
        └─> depends: ALL (Interface, SignalGenerators, DataContext, ModelsFunction, Tester)
```

**Порядок сборки:**
1. Interface
2. SignalGenerators, ModelsFunction (параллельно)
3. DataContext
4. Tester
5. MainProgram

---

## 3. Детальный план по модулям

### 3.1 Модуль: Interface/

**Назначение:** Базовые интерфейсы и структуры данных для всех модулей

**Файлы:**
```
Interface/
├── include/
│   ├── signal_data.h          (~150 строк)
│   ├── spectral_data.h        (~100 строк)
│   ├── igpu_processor.h       (~80 строк)
│   └── common_types.h         (~50 строк)
├── CMakeLists.txt             (~30 строк)
└── README.md                  (документация)
```

**Детали реализации:**

#### signal_data.h
```cpp
#pragma once
#include <vector>
#include <complex>

namespace CudaCalc {

// Конфигурация строба
struct StrobeConfig {
    int ray_count;          // 4
    int points_per_ray;     // 1024
    int window_fft;         // 16
    
    int total_points() const { return ray_count * points_per_ray; }
    int num_windows() const { return total_points() / window_fft; }
};

// Входной сигнал (HOST memory)
struct InputSignalData {
    std::vector<std::complex<float>> signal;  // 4096 точек
    StrobeConfig config;
    bool return_for_validation;  // Флаг сохранения для Python валидации
};

// Выходные спектральные данные
struct OutputSpectralData {
    // windows[окно][спектр]: 256 окон × 16 спектров
    std::vector<std::vector<std::complex<float>>> windows;
};

} // namespace CudaCalc
```

**Оценка времени:** 2-3 часа

---

### 3.2 Модуль: SignalGenerators/

**Назначение:** Генерация тестовых сигналов различных типов

**Файлы:**
```
SignalGenerators/
├── include/
│   ├── signal_types.h         (~40 строк) - enum SignalType
│   ├── base_generator.h       (~60 строк) - базовый класс
│   └── sine_generator.h       (~100 строк) - генератор синусоид
├── src/
│   ├── base_generator.cpp     (~80 строк)
│   └── sine_generator.cpp     (~150 строк)
├── tests/
│   └── test_sine_generator.cpp (~200 строк) - unit тесты
├── CMakeLists.txt             (~40 строк)
└── README.md
```

**Алгоритм SineGenerator:**

```cpp
InputSignalData SineGenerator::generate(int window_fft, bool return_for_validation) {
    InputSignalData data;
    data.config = {ray_count_, points_per_ray_, window_fft};
    data.return_for_validation = return_for_validation;
    
    int total = data.config.total_points();
    data.signal.resize(total);
    
    // Генерация синуса:
    // signal[n] = amplitude * exp(i * 2π * n / period)
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

**Параметры для FFT16 теста:**
- ray_count = 4
- points_per_ray = 1024
- period = 8 (половина окна FFT16)
- amplitude = 1.0
- phase = 0.0

**Оценка времени:** 4-5 часов

---

### 3.3 Модуль: ModelsFunction/ - FFT16 реализации

**Назначение:** Две реализации FFT16 для сравнения производительности

#### 3.3.1 FFT16_Shared2D (начинаем с этого - проще!)

**Файлы:**
```
ModelsFunction/src/nvidia/fft/FFT16_Shared2D/
├── fft16_shared2d.h           (~120 строк)
├── fft16_shared2d.cpp         (~180 строк)
├── fft16_shared2d_kernel.cu   (~250 строк)
└── CMakeLists.txt
```

**Алгоритм kernel:**

```cuda
__global__ void fft16_shared2d_kernel(
    const cuComplex* input,
    cuComplex* output,
    int num_windows
) {
    // === КОНФИГУРАЦИЯ ===
    // Block: 1024 threads = 64 FFT × 16 threads каждый
    int block_fft_id = threadIdx.x / 16;  // 0..63
    int point_id = threadIdx.x % 16;       // 0..15
    int global_fft_id = blockIdx.x * 64 + block_fft_id;
    
    if (global_fft_id >= num_windows) return;
    
    // === SHARED MEMORY 2D ===
    __shared__ float2 shmem[64][16];  // [FFT][точка]
    
    // === LOAD ===
    int input_idx = global_fft_id * 16 + point_id;
    shmem[block_fft_id][point_id] = make_float2(
        input[input_idx].x,
        input[input_idx].y
    );
    __syncthreads();
    
    // === ЛИНЕЙНАЯ РАСКРУТКА 4 STAGES ===
    
    // STAGE 0: step=1, pairs separated by 1
    {
        if (point_id < 8) {
            int idx1 = point_id * 2;
            int idx2 = idx1 + 1;
            
            float2 a = shmem[block_fft_id][idx1];
            float2 b = shmem[block_fft_id][idx2];
            
            // Twiddle factor: W_2^k = exp(-i*2π*k/2)
            float angle = -M_PI * point_id;
            float2 twiddle = make_float2(cosf(angle), sinf(angle));
            
            // Complex multiply: b * twiddle
            float2 b_tw = make_float2(
                b.x * twiddle.x - b.y * twiddle.y,
                b.x * twiddle.y + b.y * twiddle.x
            );
            
            // Butterfly
            shmem[block_fft_id][idx1] = make_float2(a.x + b_tw.x, a.y + b_tw.y);
            shmem[block_fft_id][idx2] = make_float2(a.x - b_tw.x, a.y - b_tw.y);
        }
        __syncthreads();
    }
    
    // STAGE 1: step=2, pairs separated by 2
    {
        if (point_id < 8) {
            int group = point_id / 2;
            int pos = point_id % 2;
            int idx1 = group * 4 + pos;
            int idx2 = idx1 + 2;
            
            float2 a = shmem[block_fft_id][idx1];
            float2 b = shmem[block_fft_id][idx2];
            
            float angle = -M_PI * pos / 2.0f;
            float2 twiddle = make_float2(cosf(angle), sinf(angle));
            
            float2 b_tw = make_float2(
                b.x * twiddle.x - b.y * twiddle.y,
                b.x * twiddle.y + b.y * twiddle.x
            );
            
            shmem[block_fft_id][idx1] = make_float2(a.x + b_tw.x, a.y + b_tw.y);
            shmem[block_fft_id][idx2] = make_float2(a.x - b_tw.x, a.y - b_tw.y);
        }
        __syncthreads();
    }
    
    // STAGE 2: step=4, pairs separated by 4
    {
        if (point_id < 8) {
            int group = point_id / 4;
            int pos = point_id % 4;
            int idx1 = group * 8 + pos;
            int idx2 = idx1 + 4;
            
            float2 a = shmem[block_fft_id][idx1];
            float2 b = shmem[block_fft_id][idx2];
            
            float angle = -M_PI * pos / 4.0f;
            float2 twiddle = make_float2(cosf(angle), sinf(angle));
            
            float2 b_tw = make_float2(
                b.x * twiddle.x - b.y * twiddle.y,
                b.x * twiddle.y + b.y * twiddle.x
            );
            
            shmem[block_fft_id][idx1] = make_float2(a.x + b_tw.x, a.y + b_tw.y);
            shmem[block_fft_id][idx2] = make_float2(a.x - b_tw.x, a.y - b_tw.y);
        }
        __syncthreads();
    }
    
    // STAGE 3: step=8, pairs separated by 8
    {
        if (point_id < 8) {
            int idx1 = point_id;
            int idx2 = idx1 + 8;
            
            float2 a = shmem[block_fft_id][idx1];
            float2 b = shmem[block_fft_id][idx2];
            
            float angle = -M_PI * point_id / 8.0f;
            float2 twiddle = make_float2(cosf(angle), sinf(angle));
            
            float2 b_tw = make_float2(
                b.x * twiddle.x - b.y * twiddle.y,
                b.x * twiddle.y + b.y * twiddle.x
            );
            
            shmem[block_fft_id][idx1] = make_float2(a.x + b_tw.x, a.y + b_tw.y);
            shmem[block_fft_id][idx2] = make_float2(a.x - b_tw.x, a.y - b_tw.y);
        }
        __syncthreads();
    }
    
    // === FFT SHIFT в kernel ===
    // Стандартный порядок: [0,1,2,...,7,8,-7,-6,...,-1]
    // После shift: [-8,-7,...,-1,0,1,...,7]
    int shifted_idx;
    if (point_id < 8) {
        shifted_idx = point_id + 8;  // 0→8, 1→9, ..., 7→15
    } else {
        shifted_idx = point_id - 8;  // 8→0, 9→1, ..., 15→7
    }
    
    // === STORE ===
    int output_idx = global_fft_id * 16 + shifted_idx;
    float2 result = shmem[block_fft_id][point_id];
    output[output_idx] = make_cuComplex(result.x, result.y);
}
```

**Параметры запуска:**
```cpp
int num_blocks = (256 + 63) / 64;  // 4 блока
int threads_per_block = 1024;       // 64 FFT × 16 threads
size_t shared_mem = 64 * 16 * sizeof(float2);  // 8 KB

fft16_shared2d_kernel<<<num_blocks, threads_per_block, shared_mem>>>(
    d_input, d_output, 256
);
```

**Оценка времени:** 6-8 часов

---

#### 3.3.2 FFT16_WMMA (Tensor Cores - сложнее!)

**Файлы:**
```
ModelsFunction/src/nvidia/fft/FFT16_WMMA/
├── fft16_wmma.h               (~130 строк)
├── fft16_wmma.cpp             (~200 строк)
├── fft16_wmma_kernel.cu       (~350 строк) - сложнее!
└── CMakeLists.txt
```

**Особенности:**
- Использование `wmma` namespace
- FP16 преобразования (__float2half)
- Тензорные операции через `wmma::mma_sync`
- Организация данных в 16×16 блоки для Tensor Cores

**Алгоритм (упрощённо):**
```cuda
__global__ void fft16_wmma_kernel(...) {
    // Преобразование в FP16
    __shared__ __half2 shmem[64][16];
    
    // Загрузка в half precision
    shmem[...] = __floats2half2_rn(input.x, input.y);
    
    // Butterfly через wmma (4 stages линейно)
    // ... аналогично Shared2D, но с half arithmetic ...
    
    // FFT shift
    // Store обратно в FP32
    output[...] = make_cuComplex(__half2float(...));
}
```

**Оценка времени:** 8-10 часов (сложнее из-за wmma)

---

### 3.4 Модуль: DataContext/

**Назначение:** Управление данными, логирование, архивирование

**Файлы:**
```
DataContext/
├── include/
│   ├── data_manager.h         (~120 строк)
│   ├── json_logger.h          (~100 строк)
│   ├── model_archiver.h       (~150 строк) - КРИТИЧНО!
│   └── config.h               (~60 строк)
├── src/
│   ├── data_manager.cpp       (~200 строк)
│   ├── json_logger.cpp        (~250 строк)
│   ├── model_archiver.cpp     (~300 строк) - КРИТИЧНО!
│   └── config.cpp             (~80 строк)
├── Config/
│   └── validation_params.json (~50 строк)
├── Reports/                    (создаётся автоматически)
├── ValidationData/             (создаётся автоматически)
│   └── FFT16/
├── Models/                     🔴 КРИТИЧНО!
│   └── NVIDIA/
│       └── FFT/
│           └── 16/
│               └── (версии создаются автоматически)
├── CMakeLists.txt
└── README.md
```

#### 3.4.1 JSONLogger

**Алгоритм:**
```cpp
class JSONLogger {
public:
    bool save_validation_data(
        const std::string& algorithm,
        const InputSignalData& input,
        const OutputSpectralData& output,
        const BasicProfilingResult& profiling
    ) {
        // 1. Формирование JSON
        nlohmann::json j;
        j["metadata"] = {
            {"date", get_current_date()},
            {"time", get_current_time()},
            {"gpu_model", get_gpu_name()},
            {"cuda_version", "13.0"},
            {"algorithm", algorithm}
        };
        
        j["test_config"] = {
            {"ray_count", input.config.ray_count},
            {"points_per_ray", input.config.points_per_ray},
            {"window_fft", input.config.window_fft}
        };
        
        // 2. Входной сигнал
        j["input_signal"]["real"] = extract_real(input.signal);
        j["input_signal"]["imag"] = extract_imag(input.signal);
        
        // 3. Результаты GPU
        j["gpu_results"]["num_windows"] = output.windows.size();
        for (size_t i = 0; i < output.windows.size(); ++i) {
            j["gpu_results"]["windows"][i] = {
                {"window_id", i},
                {"spectrum_real", extract_real(output.windows[i])},
                {"spectrum_imag", extract_imag(output.windows[i])}
            };
        }
        
        // 4. Профилирование
        j["profiling"] = {
            {"upload_ms", profiling.upload_ms},
            {"compute_ms", profiling.compute_ms},
            {"download_ms", profiling.download_ms},
            {"total_ms", profiling.total_ms}
        };
        
        // 5. Сохранение
        std::string filename = generate_filename(algorithm);
        std::ofstream file("DataContext/ValidationData/FFT16/" + filename);
        file << j.dump(2);  // Pretty print with indent=2
        
        return true;
    }
};
```

**Оценка времени:** 5-6 часов

---

#### 3.4.2 🔴 ModelArchiver (КРИТИЧНО!)

**Алгоритм:**

```cpp
class ModelArchiver {
private:
    std::filesystem::path base_path_;  // "DataContext/Models"
    
public:
    std::string get_next_version(
        const std::string& gpu_type,   // "NVIDIA"
        const std::string& algorithm,  // "FFT"
        int size                       // 16
    ) {
        // 1. Путь к моделям
        auto path = base_path_ / gpu_type / algorithm / std::to_string(size);
        
        // 2. Поиск существующих версий
        int max_version = 0;
        if (std::filesystem::exists(path)) {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                std::string dirname = entry.path().filename().string();
                // Парсинг "model_2025_10_09_v3" → v3
                if (dirname.find("_v") != std::string::npos) {
                    int v = parse_version(dirname);
                    max_version = std::max(max_version, v);
                }
            }
        }
        
        // 3. Следующая версия
        return "model_" + get_date() + "_v" + std::to_string(max_version + 1);
    }
    
    bool save_model(
        const ModelInfo& info,
        const std::vector<std::string>& source_files,  // ["fft16_wmma.cu", ...]
        const std::string& results_json,
        const std::string& description
    ) {
        // 1. Создание директории
        auto model_path = base_path_ / info.gpu_type / info.algorithm / 
                         std::to_string(info.size) / info.version;
        std::filesystem::create_directories(model_path);
        
        // 2. Копирование исходников
        for (const auto& src : source_files) {
            std::filesystem::copy_file(
                src,
                model_path / std::filesystem::path(src).filename()
            );
        }
        
        // 3. Сохранение results.json
        std::ofstream(model_path / "results.json") << results_json;
        
        // 4. Сохранение описания
        std::ofstream(model_path / "description.txt") << description;
        
        std::cout << "✅ Model saved: " << model_path << std::endl;
        return true;
    }
};
```

**Использование:**
```cpp
// После теста FFT16_WMMA
ModelArchiver archiver;
ModelInfo info{
    .gpu_type = "NVIDIA",
    .algorithm = "FFT",
    .size = 16,
    .version = archiver.get_next_version("NVIDIA", "FFT", 16),  // auto: v1, v2, ...
    .description = "Baseline FFT16 с Tensor Cores"
};

archiver.save_model(
    info,
    {"ModelsFunction/src/nvidia/fft/FFT16_WMMA/fft16_wmma.cu",
     "ModelsFunction/src/nvidia/fft/FFT16_WMMA/fft16_wmma.cpp"},
    profiling_json,
    "Первая реализация FFT16 через wmma"
);
```

**Оценка времени:** 6-8 часов (критичная функция!)

---

### 3.5 Модуль: Tester/

**Назначение:** Профилирование и валидация

**Файлы:**
```
Tester/
├── include/
│   ├── performance/
│   │   ├── basic_profiler.h       (~100 строк)
│   │   ├── memory_profiler.h      (~120 строк)
│   │   └── profiling_data.h       (~80 строк)
│   └── test_runner.h              (~90 строк)
├── src/
│   ├── performance/
│   │   ├── basic_profiler.cpp     (~180 строк)
│   │   └── memory_profiler.cpp    (~200 строк)
│   └── test_runner.cpp            (~150 строк)
└── CMakeLists.txt
```

#### BasicProfiler

**Алгоритм:**
```cpp
class BasicProfiler {
private:
    cudaEvent_t events_[6];  // start/end для 3 фаз
    
public:
    BasicProfiler() {
        for (int i = 0; i < 6; ++i) {
            cudaEventCreate(&events_[i]);
        }
    }
    
    void profile_upload(std::function<void()> upload_func) {
        cudaEventRecord(events_[0]);  // start_upload
        upload_func();                // cudaMemcpy H→D
        cudaEventRecord(events_[1]);  // end_upload
    }
    
    void profile_compute(std::function<void()> kernel_func) {
        cudaEventRecord(events_[2]);  // start_compute
        kernel_func();                // kernel<<<>>>()
        cudaEventRecord(events_[3]);  // end_compute
    }
    
    void profile_download(std::function<void()> download_func) {
        cudaEventRecord(events_[4]);  // start_download
        download_func();              // cudaMemcpy D→H
        cudaEventRecord(events_[5]);  // end_download
    }
    
    BasicProfilingResult get_results() {
        cudaEventSynchronize(events_[5]);
        
        float upload_ms, compute_ms, download_ms;
        cudaEventElapsedTime(&upload_ms, events_[0], events_[1]);
        cudaEventElapsedTime(&compute_ms, events_[2], events_[3]);
        cudaEventElapsedTime(&download_ms, events_[4], events_[5]);
        
        return {
            upload_ms, compute_ms, download_ms,
            upload_ms + compute_ms + download_ms,
            get_gpu_name(), "13.0", get_driver_version(),
            get_timestamp()
        };
    }
};
```

**Оценка времени:** 4-5 часов

---

### 3.6 Модуль: MainProgram/

**Назначение:** Точка входа, интеграция всех модулей

**Файлы:**
```
MainProgram/
├── src/
│   └── main_fft16_test.cpp    (~300 строк)
└── CMakeLists.txt
```

**Алгоритм main_fft16_test.cpp:**

```cpp
#include "SignalGenerators/sine_generator.h"
#include "ModelsFunction/nvidia/fft/fft16_wmma.h"
#include "ModelsFunction/nvidia/fft/fft16_shared2d.h"
#include "Tester/performance/basic_profiler.h"
#include "DataContext/json_logger.h"
#include "DataContext/model_archiver.h"

int main(int argc, char** argv) {
    std::cout << "=== FFT16 Baseline Test ===" << std::endl;
    
    // === 1. ГЕНЕРАЦИЯ СИГНАЛА ===
    SineGenerator generator(4, 1024, 8);  // 4 луча, 1024 точки, период 8
    auto input = generator.generate(16, true);  // wFFT=16, return_for_validation=true
    
    std::cout << "✓ Signal generated: " << input.signal.size() << " points" << std::endl;
    
    // === 2. ТЕСТ FFT16_WMMA ===
    {
        std::cout << "\n--- Testing FFT16_WMMA ---" << std::endl;
        
        FFT16_WMMA fft_wmma;
        fft_wmma.initialize();
        
        BasicProfiler profiler;
        OutputSpectralData output;
        
        // Профилирование
        profiler.profile_upload([&]() {
            fft_wmma.upload_input(input);
        });
        
        profiler.profile_compute([&]() {
            fft_wmma.execute();
        });
        
        profiler.profile_download([&]() {
            output = fft_wmma.download_output();
        });
        
        auto prof_result = profiler.get_results();
        
        std::cout << "  Upload:   " << prof_result.upload_ms << " ms" << std::endl;
        std::cout << "  Compute:  " << prof_result.compute_ms << " ms" << std::endl;
        std::cout << "  Download: " << prof_result.download_ms << " ms" << std::endl;
        std::cout << "  TOTAL:    " << prof_result.total_ms << " ms" << std::endl;
        
        // Сохранение для Python валидации
        if (input.return_for_validation) {
            JSONLogger logger;
            logger.save_validation_data("FFT16_WMMA", input, output, prof_result);
            std::cout << "✓ Saved to ValidationData/" << std::endl;
        }
        
        // 🔴 АРХИВИРОВАНИЕ МОДЕЛИ
        ModelArchiver archiver;
        ModelInfo model_info{
            .gpu_type = "NVIDIA",
            .algorithm = "FFT",
            .size = 16,
            .version = archiver.get_next_version("NVIDIA", "FFT", 16)
        };
        archiver.save_model(
            model_info,
            {"ModelsFunction/src/nvidia/fft/FFT16_WMMA/fft16_wmma.cu",
             "ModelsFunction/src/nvidia/fft/FFT16_WMMA/fft16_wmma.cpp"},
            prof_result.to_json(),
            "Baseline FFT16 WMMA implementation"
        );
        std::cout << "✓ Model archived: " << model_info.version << std::endl;
        
        fft_wmma.cleanup();
    }
    
    // === 3. ТЕСТ FFT16_Shared2D ===
    {
        std::cout << "\n--- Testing FFT16_Shared2D ---" << std::endl;
        // ... аналогично WMMA ...
    }
    
    // === 4. СРАВНЕНИЕ ===
    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "Run Python validator:" << std::endl;
    std::cout << "  cd Validator && python validate_fft.py" << std::endl;
    
    return 0;
}
```

**Оценка времени:** 4-5 часов

---

## 4. CMake Build System

### 4.1 Корневой CMakeLists.txt

**Файл:** `CMakeLists.txt` (~100 строк)

```cmake
cmake_minimum_required(VERSION 3.20)
project(CudaCalc VERSION 0.1.0 LANGUAGES CXX CUDA)

# === C++ STANDARD ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# === CUDA SETUP ===
enable_language(CUDA)
find_package(CUDAToolkit 13.0 REQUIRED)

# Compute Capability для RTX 3060
set(CMAKE_CUDA_ARCHITECTURES "86")  # Ampere

# === DEPENDENCIES ===
# JSON library
find_package(nlohmann_json 3.11.0 REQUIRED)

# Google Test (опционально)
find_package(GTest)

# === COMPILER FLAGS ===
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")

# Release оптимизации
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Debug информация
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

# === INCLUDE PATHS ===
include_directories(${CMAKE_SOURCE_DIR})

# === SUBDIRECTORIES ===
add_subdirectory(Interface)
add_subdirectory(SignalGenerators)
add_subdirectory(DataContext)
add_subdirectory(ModelsFunction)
add_subdirectory(Tester)
add_subdirectory(MainProgram)

# === INSTALL (опционально) ===
install(TARGETS cudacalc_main DESTINATION bin)
```

**Оценка времени:** 3-4 часа (с отладкой)

---

### 4.2 Пример CMakeLists.txt для модуля

**ModelsFunction/CMakeLists.txt:**
```cmake
add_library(ModelsFunction STATIC)

# Исходники
target_sources(ModelsFunction
    PRIVATE
        src/nvidia/fft/FFT16_Shared2D/fft16_shared2d.cpp
        src/nvidia/fft/FFT16_Shared2D/fft16_shared2d_kernel.cu
        src/nvidia/fft/FFT16_WMMA/fft16_wmma.cpp
        src/nvidia/fft/FFT16_WMMA/fft16_wmma_kernel.cu
)

# Заголовки
target_include_directories(ModelsFunction
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Зависимости
target_link_libraries(ModelsFunction
    PUBLIC
        Interface
        CUDA::cufft
        CUDA::cudart
)

# CUDA настройки
set_target_properties(ModelsFunction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)
```

---

## 5. План тестирования

### 5.1 Unit Tests

**Что тестируем:**
- SineGenerator: корректность генерации
- FFT16 kernels: небольшие известные сигналы
- JSONLogger: валидность JSON
- ModelArchiver: создание/чтение директорий

**Фреймворк:** Google Test

### 5.2 Integration Test

**Сценарий:**
```
1. Генерация сигнала
2. FFT16_Shared2D
3. FFT16_WMMA
4. Сохранение JSON
5. Архивирование моделей
6. Python валидация
```

### 5.3 Performance Test

**Метрики:**
- Upload time < 0.5 ms
- Compute time < 1.0 ms (target)
- Download time < 0.5 ms
- Total < 2.0 ms

---

## 6. Последовательность реализации (фазы)

### 📌 Фаза 1: Базовая инфраструктура (3-4 дня)

**День 1-2:**
- [ ] CMakeLists.txt корневой
- [ ] Interface/ модуль (headers)
- [ ] CMake для каждого модуля
- [ ] Первая успешная компиляция (пустые модули)

**День 3:**
- [ ] SignalGenerators/ структура
- [ ] SineGenerator базовая реализация
- [ ] Unit тесты для SineGenerator

**День 4:**
- [ ] DataContext/ структура
- [ ] JSONLogger базовая реализация
- [ ] Создание директорий Models/, Reports/, ValidationData/

**Критерий завершения:**
✅ Проект компилируется
✅ SineGenerator генерирует корректный сигнал
✅ JSONLogger сохраняет простой JSON

---

### 📌 Фаза 2: FFT16 реализация (4-5 дней)

**День 5-6: FFT16_Shared2D**
- [ ] Kernel реализация
- [ ] Линейная раскрутка 4 stages
- [ ] FFT shift в kernel
- [ ] Wrapper класс
- [ ] Базовый тест (известный сигнал)

**День 7-8: FFT16_WMMA**
- [ ] Kernel с wmma
- [ ] FP16 преобразования
- [ ] Линейная раскрутка
- [ ] Wrapper класс
- [ ] Базовый тест

**День 9:**
- [ ] Отладка обеих реализаций
- [ ] Проверка корректности (визуальная)

**Критерий завершения:**
✅ Обе реализации компилируются
✅ Выдают разумные результаты (без NaN/Inf)

---

### 📌 Фаза 3: Профилирование (1-2 дня)

**День 10:**
- [ ] BasicProfiler (cudaEvent)
- [ ] Интеграция в main
- [ ] Вывод результатов в консоль
- [ ] Базовые метрики получены

**День 11 (опционально):**
- [ ] MemoryProfiler
- [ ] Расширенные метрики

**Критерий завершения:**
✅ Профилирование работает
✅ Метрики логируются в JSON

---

### 📌 Фаза 4: 🔴 ModelArchiver (1-2 дня)

**День 11-12:**
- [ ] ModelArchiver класс
- [ ] Автоматическое версионирование
- [ ] Копирование исходников
- [ ] Сохранение results.json
- [ ] Интеграция в main
- [ ] Проверка что Models/ создаются

**Критерий завершения:**
✅ После каждого теста создаётся новая версия
✅ Исходники сохранены
✅ Результаты сохранены
✅ НЕТ перезаписи!

---

### 📌 Фаза 5: Интеграция и тестирование (1-2 дня)

**День 13:**
- [ ] Полная интеграция всех модулей
- [ ] End-to-end тест
- [ ] Python валидация
- [ ] Проверка корректности через scipy.fft

**День 14:**
- [ ] Сравнение WMMA vs Shared2D
- [ ] Выбор лучшего варианта
- [ ] Документирование результатов
- [ ] Обновление CLAUDE.md

**Критерий завершения:**
✅ Вся цепочка работает
✅ Результаты корректны (error < 0.01%)
✅ Определен fastest algorithm
✅ Baseline метрики задокументированы

---

## 7. Риски и митигация

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| CMake проблемы на Ubuntu | Средняя | Использовать стандартные пути, тестировать рано |
| wmma сложность | Высокая | Начать с Shared2D, изучить примеры NVIDIA |
| FP16 потеря точности | Высокая | Валидация покажет, если > 0.01% → используем Shared2D |
| Недооценка времени | Средняя | Буфер +20% на каждую фазу |
| Ошибки в butterfly | Высокая | Линейная раскрутка, проверка на простых сигналах |

---

## 8. Критерии успеха

### 8.1 Функциональные
- ✅ Обе реализации FFT16 работают
- ✅ Профилирование корректное
- ✅ Валидация проходит (error < 0.01%)
- ✅ ModelArchiver сохраняет модели
- ✅ Python validator работает

### 8.2 Нефункциональные
- ✅ Compute time измерен
- ✅ Fastest algorithm определён
- ✅ Code coverage >= 70% (минимум)
- ✅ Нет memory leaks (cuda-memcheck)

### 8.3 Документация
- ✅ plan.md заполнен
- ✅ tasks.md создан
- ✅ CLAUDE.md обновлён с результатами
- ✅ MemoryBank содержит ключевые решения

---

## 9. Следующие шаги после завершения

1. Создать spec для FFT32 (specs/002-fft32-implementation/)
2. Портировать лучшее решение (WMMA или Shared2D)
3. Продолжить по ROADMAP.md

---

## Приложения

### A. Полезные команды

**Компиляция:**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Запуск:**
```bash
./MainProgram/cudacalc_fft16_test
```

**Python валидация:**
```bash
cd Validator
python validate_fft.py
```

**Профилирование:**
```bash
ncu --metrics=all ./cudacalc_fft16_test
```

---

**Статус:** ✅ Готов к реализации  
**Версия:** 1.0  
**Автор:** AlexLan73  
**Дата:** 10 октября 2025

